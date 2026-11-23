# -*- coding: utf-8 -*-
"""决策 prompt 的构造逻辑，从 EasyPlanPlugin 里原样搬出来。

搬出来的唯一目的：让探针实验能调用**和仿真完全相同**的 prompt 构造代码。
如果探针自己复刻一份 prompt，那测的就是复刻件，不是被试系统。

这个模块刻意不依赖 agentkernel_standalone —— 只依赖 role_config 里的几个
纯字典。所以不起仿真、不连 LLM 也能 import。

EasyPlanPlugin 从这里 import 并薄封装，两边共用同一份定义。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from role_config import ROLE_ACTIONS, ROLE_ACTION_LABEL, ROLE_SYSTEM_PROMPTS


LOCATION_NAMES_ZH = {
    "dorm_zone": "学生宿舍", "staff_dorm": "教职工宿舍",
    "campus_gate": "校门口",
    "library": "图书馆", "teaching_1": "东教学楼",
    "teaching_2": "西教学楼", "cafeteria_1": "东食堂", "cafeteria_2": "西食堂",
    "sports_field": "运动场", "main_road": "校园主路",
}

CLASS_SCHEDULES = {
    "A1": [1, 2, 7, 8],
    "A2": [1, 3, 7, 9],
    "A3": [2, 3, 8, 9],
    "A4": [1, 4, 7, 8],
    "A5": [2, 4, 8, 9],
    "A6": [3, 4, 7, 9],
}

VALID_ACTIONS = {"study", "exercise", "rest", "eat", "wander"}  # 学生自由行为（回退用）


def get_system_prompt(role: str) -> str:
    """根据角色获取对应的 SYSTEM_PROMPT"""
    return ROLE_SYSTEM_PROMPTS.get(role, ROLE_SYSTEM_PROMPTS["student"])

def get_valid_actions(role: str) -> set:
    """根据角色获取可用行为集"""
    return ROLE_ACTIONS.get(role, ROLE_ACTIONS["student"])

def get_role_examples(role: str) -> str:
    """返回角色对应的决策示例"""
    if role == "student":
        return (
            "示例1：早上8点，有课，精力0.8 → go_class\n"
            "示例2：中午12点，饥饿0.3，精力0.6 → eat\n"
            "示例3：晚上10点，精力0.3 → rest\n"
            "示例4：下午3点，精力0.7，自律性0.7 → study\n"
            "示例5：下午5点，精力0.8，运动爱好0.6 → exercise\n"
            "示例6：早上7点，精力0.9 → rest（还想多睡一会儿）\n"
            "示例7：晚上8点，社交需求0.3 → chat\n"
            "注意：真实大学生约30%时间在宿舍休息、25%上课、15%学习、10%吃饭、10%运动、10%其他。\n"
            "不要总是选择学习——要根据自己的性格和状态做出真实的选择。"
        )
    elif role == "teacher":
        return (
            "示例1：早上8点，有课，精力0.8 → teach_class\n"
            "示例2：上午10点，无课，精力0.7 → prepare_lesson\n"
            "示例3：下午3点，无课，有作业待批 → grade\n"
            "示例4：中午12点，饥饿0.3 → eat"
        )
    elif role == "cafeteria_staff":
        return (
            "示例1：早上6点，工作时间 → cook\n"
            "示例2：中午12点，饭点 → serve\n"
            "示例3：下午1点，饭后 → clean_kitchen\n"
            "示例4：下午3点，非工作时段，精力0.3 → rest"
        )
    elif role == "librarian":
        return (
            "示例1：早上8点，工作时间 → manage_books\n"
            "示例2：上午10点，有读者咨询 → help_reader\n"
            "示例3：下午2点，工作时间 → patrol_library\n"
            "示例4：中午12点，饥饿0.3 → eat"
        )
    return ""

def build_llm_prompt(payload, hour, hour_in_day, day_type, weather_zh, has_duty, recent_str, p, memory_context: str = "", peer_info: str = "") -> str:
    """构建"原则+示例"模式的prompt（第一层：软约束提示），根据角色动态生成"""
    role = p.role
    hour_str = f"{hour}:00"

    # 工作/上课提醒
    duty_reminder = ""
    if has_duty:
        if role == "student":
            duty_reminder = "\n【当前有课提醒】现在是上课时间，你可以考虑去上课。"
        elif role == "teacher":
            duty_reminder = "\n【当前有课提醒】现在是你的授课时间，你应该去教室上课。"
        elif role == "cafeteria_staff":
            duty_reminder = "\n【当前是工作时间】现在是食堂工作时间，你应该在岗位上工作。"
        elif role == "librarian":
            duty_reminder = "\n【当前是工作时间】现在是图书馆开放时间，你应该在岗位上工作。"

    # 根据角色构建可用行为列表
    role_actions = ROLE_ACTIONS.get(role, ROLE_ACTIONS["student"])
    actions_text = "\n".join(
        f"- {a}：{ROLE_ACTION_LABEL.get(a, a)}" for a in sorted(role_actions)
    )

    # 角色特有信息
    role_extra_info = ""
    if role == "student":
        role_extra_info = f"课程组：{p.class_group}\n"
    elif role == "teacher":
        role_extra_info = f"任教课程组：{', '.join(p.teach_classes)}\n工作地点：{p.workplace}\n"
    elif role == "cafeteria_staff":
        role_extra_info = f"工作地点：{p.workplace}\n"
    elif role == "librarian":
        role_extra_info = f"工作地点：{p.workplace}\n"

    # 根据角色构建示例
    role_examples = get_role_examples(role)

    return f'''【社会行为参考原则】
1. 生存优先：当你非常饿或非常累时，应该优先满足基本需求
2. 责任意识：在合适的时间完成应该做的事（工作、上课等）
3. 社交平衡：适当与他人交流，但不要过度
4. 劳逸结合：学习和休息要平衡，不要一直学习
5. 你的行为应该符合你的身份、性格和目标
6. 真实感：真实的大学生不会一整天都在学习——他们也会赖床、刷手机、运动、发呆、和朋友聊天
7. 时段感：清晨和深夜应该休息；饭点应该吃饭；白天上课和学习
8. 你身处宿舍中，可以感知到同伴的行为和学习氛围。这些信息是你做决策时可参考的环境因素之一，但你最终的选择应该基于自己的身份、性格和目标

【你可以做的事情】
{actions_text}

【你的个人信息】
姓名：{payload['name']}
身份：{payload['identity']}
性格：{payload['personality_desc']}
习惯：{payload['habits']}
长期目标：{payload['goal']}
{role_extra_info}自律性（0-1）：{p.conscientiousness:.2f}
动力（0-1）：{p.motivation:.2f}

【当前环境】
时间：{hour_str}
星期类型：{day_type}
天气：{weather_zh}
{duty_reminder}

【当前状态】
精力：{payload['current_state']['energy']:.2f}
饱腹：{payload['current_state']['hunger']:.2f}
社交需求：{payload['current_state']['social']:.2f}
心情：{payload['current_state']['mood']:.2f}
当前位置：{p.current_location if hasattr(p, 'current_location') else '未知'}
{recent_str}
{memory_context}
{peer_info}
【示例】
{role_examples}

请做出你认为最合适的选择，并简单说明理由。
'''

def build_peer_info_block(peer_rows, cumulative):
    """同伴信息 prompt 片段。

    peer_rows: [(name, action, location), ...]  —— 已经解析好的同伴状态
    cumulative: peer_study_cumulative 当前值

    纯函数：不碰 agent_manager，不读 profile。EasyPlanPlugin 负责把活的
    peer 对象翻译成 peer_rows 再调这里，探针则直接喂构造好的 rows。
    """
    if not peer_rows:
        return ""

    action_zh = ROLE_ACTION_LABEL
    loc_zh = LOCATION_NAMES_ZH
    peer_lines = []
    for name, action, location in peer_rows:
        loc_name = loc_zh.get(location, location)
        act_name = action_zh.get(action, action)
        peer_lines.append(f"{name} · {loc_name} · {act_name}")

    peer_block = "\n".join(peer_lines)

    return (
        f"\n【宿舍同伴当前动态】\n"
        f"{peer_block}\n"
        f"\n【宿舍整体学习氛围】\n"
        f"近期同伴平均学习强度: {cumulative:.2f} (0-1 范围)\n"
        f"\n以上信息仅供参考，你可以根据自己的身份、性格和目标，\n"
        f"自行判断是否受其影响。\n"
    )


def check_has_duty(role, hour_in_day, is_weekday, class_group=None,
                   teach_schedule=(), shift=()):
    """是否处于「必须上课/工作」时段。与 EasyPlanPlugin._check_has_duty 同构。"""
    if role == "student":
        return hour_in_day in CLASS_SCHEDULES.get(class_group, []) and is_weekday
    elif role == "teacher":
        return hour_in_day in teach_schedule and is_weekday
    elif role in ("cafeteria_staff", "librarian"):
        return hour_in_day in shift
    return False
