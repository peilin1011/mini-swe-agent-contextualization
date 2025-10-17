"""Agent with workflow-based message summarization and compression.

This agent extends DefaultAgent with intelligent message compression:
- Keeps first N messages (system prompt, task description, etc.)
- Keeps last M rounds per completed sub-task
- Keeps all messages for current task
- Summarizes and compresses intermediate messages using a summary model
"""

import json
import os
import re
import subprocess
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model
from minisweagent.utils.log import logger


@dataclass
class AgentConfig:
    """Configuration for agent with workflow compression."""
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0
    
    # Workflow compression settings
    enable_condenser: bool = True
    keep_first: int = 4
    keep_last_round_per_task: int = 1
    condenser_template: str = ""
    summary_model: dict[str, Any] | None = field(default_factory=lambda: None)
    instance_name: str = ""
    
    # Compression logging settings
    enable_compression_logging: bool = True
    compression_log_dir: str = "compression_logs"


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.model = model
        self.env = env
        self.messages = []
        self.summary_model = None
        
        # 添加summary model统计跟踪
        self.summary_model_cost = 0.0
        self.summary_model_calls = 0

        # 初始化压缩日志目录（如果启用记录）
        if self.config.enable_compression_logging:
            self.compression_log_dir = Path(self.config.compression_log_dir)
            self.compression_log_dir.mkdir(exist_ok=True)
            
            # 为每个instance创建日志文件
            if self.config.instance_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.compression_log_file = self.compression_log_dir / f"{self.config.instance_name}_{timestamp}.jsonl"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                self.compression_log_file = self.compression_log_dir / f"compression_{timestamp}.jsonl"
        else:
            self.compression_log_dir = None
            self.compression_log_file = None

        # 只在启用 condenser 时才初始化总结模型
        if not self.config.enable_condenser:
            logger.info("ℹ️  工作流程压缩功能已禁用 (enable_condenser=False)")
            return

        # 初始化本地总结模型 - 从config的summary_model节点读取配置
        if self.config.summary_model and self.config.summary_model.get("model_name"):
            try:
                from minisweagent.models import get_model

                summary_model_config = {
                    "model_name": self.config.summary_model["model_name"],
                    "model_kwargs": {
                        "api_base": self.config.summary_model.get("api_base", ""),
                        "api_key": self.config.summary_model.get("api_key", ""),
                        "temperature": 0.0,
                        "drop_params": True,
                    }
                }

                self.summary_model = get_model(
                    self.config.summary_model["model_name"],
                    config=summary_model_config
                )
                logger.info("✅ 工作流程压缩功能已启用")
                logger.info(f"   总结模型: {self.config.summary_model['model_name']}")
            except Exception as e:
                logger.error(f"⚠️ 总结模型初始化失败，压缩功能将被禁用: {e}")
                import traceback
                traceback.print_exc()
                self.summary_model = None
        else:
            logger.warning("⚠️ enable_condenser=True 但未配置 summary_model.model_name，压缩功能将被禁用")

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def _log_compression_interaction(self, input_conversation: str, output_summary: str, 
                                   model_name: str, step_count: int, message_count: int):
        """记录压缩LLM的输入输出交互"""
        # 检查是否启用压缩日志记录
        if not self.config.enable_compression_logging or not self.compression_log_file:
            return
            
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "step_count": step_count,
                "message_count": message_count,
                "model_name": model_name,
                "input": input_conversation,  # 完整的对话文本字符串
                "output": output_summary,  # 压缩生成的summary
                "instance_name": self.config.instance_name,
                # 添加统计信息
                "summary_model_calls": self.summary_model_calls,
                "summary_model_cost": self.summary_model_cost,
                "main_model_calls": getattr(self.model, 'n_calls', 0),
                "main_model_cost": getattr(self.model, 'cost', 0.0)
            }
            
            # 追加到JSONL文件
            with open(self.compression_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
            logger.debug(f"📝 压缩交互已记录到: {self.compression_log_file}")
            logger.debug(f"📊 本次记录包含统计: {self.summary_model_calls} 次调用, ${self.summary_model_cost:.4f} 花费")
            
        except Exception as e:
            logger.error(f"⚠️ 压缩日志记录失败: {e}")

    def _should_condense_and_compress(self) -> bool:
        """使用总结模型一步完成判断和压缩 - 如果完成subtask就用condenser_template总结，否则skip"""
        # 检查是否启用压缩功能
        if not self.config.enable_condenser or not self.summary_model:
            return False
        
        # 分析全部上下文消息
        messages_to_analyze = self.messages
        # 只需要一次检查：至少需要前2条+keep_first
        min_messages_needed = 2 + self.config.keep_first
        if len(messages_to_analyze) < min_messages_needed:
            return False
        
        # 构建要分析的对话文本
        conversation_text = ""
        for i, msg in enumerate(messages_to_analyze):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if len(content) > 40000:
                content = content[:40000] + "..."
            conversation_text += f"[Message {i}] {role}:\n{content}\n\n"
        # 检查是否配置了 condenser_template
        if not self.config.condenser_template:
            logger.debug("⚠️ 未配置 condenser_template，跳过压缩")
            return False

        # 使用总结模型判断是否需要压缩并生成总结
        condenser_prompt = self.render_template(self.config.condenser_template, conversation_text=conversation_text)

        try:
            analysis_messages = [{"role": "user", "content": condenser_prompt}]
            response = self.summary_model.query(analysis_messages)
            response_content = response.get("content", "").strip()

            if "NO_COMPRESSION_NEEDED" in response_content.upper():
                logger.debug(f"🤖 {self.config.summary_model['model_name']}: 暂不压缩 - 未找到已完成的子任务")
                logger.debug(f"📊 Summary模型统计: {self.summary_model_calls} 次压缩, ${self.summary_model_cost:.4f} 花费")
                return False

            # 只有在实际压缩时才记录统计
            self.summary_model_calls += 1
            # 计算本次调用的成本（当前总成本 - 之前的总成本）
            current_total_cost = self.summary_model.cost
            previous_total_cost = self.summary_model_cost
            self.summary_model_cost = current_total_cost

            # 只有在实际压缩时才记录日志
            logger.info(f"🤖 {self.config.summary_model['model_name']}: 需要压缩 - 找到已完成的子任务")
            logger.info(f"📊 Summary模型统计: {self.summary_model_calls} 次压缩, ${self.summary_model_cost:.4f} 花费")
            logger.debug(f"📝 总结内容预览: {response_content[:1000]}...")

            # 记录压缩LLM的输入输出
            step_count = getattr(self.model, 'n_calls', 0)
            self._log_compression_interaction(
                input_conversation=conversation_text,  # 完整的对话文本字符串
                output_summary=response_content,  # 压缩生成的summary
                model_name=self.config.summary_model['model_name'],
                step_count=step_count,
                message_count=len(self.messages)
            )

            # 执行压缩，使用模型返回的总结
            return self._execute_compression_with_summary(response_content)

        except Exception as e:
            logger.error(f"⚠️ {self.config.summary_model['model_name']} 压缩失败: {e}")
            traceback.print_exc()
            return False

    def _execute_compression_with_summary(self, summary: str) -> bool:
        """执行消息压缩，将summary插入到指定位置"""
        try:
            # 1. 保留前2条：system instruction + user PR
            keep_first_two = self.messages[:2]
            
            # 2. 计算要保留的最后几轮对话
            messages_after_pr = self.messages[2:]  # system + user PR 之后的所有消息
            
            # 3. 计算最后几轮对话
            if self.config.keep_last_round_per_task > 0:
                rounds_to_keep = self.config.keep_last_round_per_task * 2
                last_rounds = messages_after_pr[-rounds_to_keep:]
            else:
                last_rounds = []
            
            # 4. 创建总结消息
            summary_msg = {
                "role": "user",
                "content": f"[Previous work summary]\n{summary}"
            }
            
            # 5. 重建消息列表：前2条 + summary + 最后几轮对话
            original_count = len(self.messages)
            self.messages = keep_first_two + [summary_msg] + last_rounds
            new_count = len(self.messages)
            
            logger.info(f"🔄 [工作流程压缩] 执行压缩: {original_count} 条消息")
            logger.info(f"✅ [工作流程压缩] 完成: {original_count} → {new_count} 条消息")
            logger.info(f"📊 压缩后消息数量: {new_count}")
            logger.info(f"📋 保留结构: 前2条 + 总结 + 最后{self.config.keep_last_round_per_task}轮")
            
            return True
            
        except Exception as e:
            logger.error(f"⚠️ 压缩执行失败: {e}")
            traceback.print_exc()
            return False

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})
        
        # 使用 debug 级别，只写文件，不显示在终端
        logger.info(f"[MESSAGE] Role: {role.upper()}")
        logger.info(f"[CONTENT] {content}")
        logger.info("-" * 80)

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars = {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        # 在每步之前检查是否需要压缩
        if self.config.enable_condenser and self.summary_model:
            self._should_condense_and_compress()
        
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))

    def get_summary_model_stats(self) -> dict[str, Any]:
        """获取summary model的统计信息（只计算实际压缩的次数）"""
        return {
            "summary_model_calls": self.summary_model_calls,
            "summary_model_cost": self.summary_model_cost,
            "summary_model_name": self.config.summary_model.get("model_name", "") if self.config.summary_model else "",
        }
