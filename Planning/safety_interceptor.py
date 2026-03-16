"""
安全拦截模块（Human-in-the-loop）
检测高风险不可逆操作，拦截并请求人类确认
"""
import logging
from typing import List

from Planning.planner import SubGoal

logger = logging.getLogger(__name__)


class SafetyInterceptor:
    """
    安全拦截器
    对涉及支付、删除、授权等高风险操作进行拦截
    通过控制台交互获取人类确认
    """

    def __init__(self, high_risk_keywords: List[str]):
        """
        :param high_risk_keywords: 高风险关键词列表
        """
        self.keywords = [kw.lower() for kw in high_risk_keywords]
        logger.info("SafetyInterceptor 初始化完成, 高风险关键词: %d 个", len(self.keywords))

    def check(self, subgoal: SubGoal) -> bool:
        """
        检查子目标是否涉及高风险操作
        如果涉及，则拦截并请求人类确认
        :param subgoal: 待检查的子目标
        :return: True = 允许执行, False = 用户拒绝
        """
        risk_matches = self._detect_risk(subgoal)

        if not risk_matches:
            logger.debug("安全检查通过: '%s'", subgoal.description)
            return True

        # 触发拦截
        logger.warning(
            "⚠️ 检测到高风险操作! 子目标='%s', 匹配关键词=%s",
            subgoal.description, risk_matches,
        )

        return self._request_human_confirmation(subgoal, risk_matches)

    def _detect_risk(self, subgoal: SubGoal) -> List[str]:
        """检测子目标中的高风险关键词"""
        matches = []
        check_text = " ".join(
            str(x) for x in [
                subgoal.description,
                subgoal.target_widget_text,
                subgoal.input_text,
                subgoal.acceptance_criteria,
            ] if x is not None
        ).lower()

        for keyword in self.keywords:
            if keyword in check_text:
                matches.append(keyword)

        return matches

    def _request_human_confirmation(self, subgoal: SubGoal, risk_keywords: List[str]) -> bool:
        """
        通过控制台请求人类确认
        :return: True = 确认执行, False = 拒绝
        """
        print("\n" + "=" * 60)
        print("🚨 安全拦截 - 检测到高风险操作")
        print("=" * 60)
        print(f"  子目标: {subgoal.description}")
        print(f"  动作类型: {subgoal.action_type}")
        print(f"  目标控件: {subgoal.target_widget_text}")
        if subgoal.input_text:
            print(f"  输入文本: {subgoal.input_text}")
        print(f"  风险关键词: {', '.join(risk_keywords)}")
        print("=" * 60)

        while True:
            try:
                response = input("是否允许执行该操作？(y/n): ").strip().lower()
                if response in ("y", "yes", "是"):
                    logger.info("用户确认执行高风险操作: '%s'", subgoal.description)
                    return True
                elif response in ("n", "no", "否"):
                    logger.info("用户拒绝执行高风险操作: '%s'", subgoal.description)
                    return False
                else:
                    print("请输入 y 或 n")
            except (EOFError, KeyboardInterrupt):
                logger.warning("用户中断输入, 默认拒绝高风险操作")
                return False
