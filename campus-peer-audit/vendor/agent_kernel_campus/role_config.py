"""
多角色配置模块
定义各角色的可用行为集、行为→地点映射、状态影响、LLM提示词等
"""

# =============================================================================
# 角色类型枚举
# =============================================================================
ROLES = ["student", "teacher", "cafeteria_staff", "librarian"]

# =============================================================================
# 各角色可用行为集
# =============================================================================
ROLE_ACTIONS = {
    "student": {"go_class", "study", "exercise", "chat", "rest", "eat", "wander"},
    "teacher": {"teach_class", "prepare_lesson", "grade", "office_hours", "chat", "rest", "eat", "wander", "meeting"},
    "cafeteria_staff": {"cook", "serve", "clean_kitchen", "rest", "eat", "chat", "wander"},
    "librarian": {"manage_books", "help_reader", "patrol_library", "rest", "eat", "chat", "wander"},
}

# =============================================================================
# 各角色行为的中文标签
# =============================================================================
ROLE_ACTION_LABEL = {
    # 学生行为
    "go_class": "上课", "study": "学习", "exercise": "运动",
    "chat": "聊天", "rest": "休息", "eat": "吃饭", "wander": "闲逛",
    # 老师行为
    "teach_class": "授课", "prepare_lesson": "备课", "grade": "批改作业",
    "office_hours": "坐班答疑", "meeting": "开会",
    # 食堂工作人员行为
    "cook": "做饭", "serve": "打饭", "clean_kitchen": "清洁厨房",
    # 图书馆管理员行为
    "manage_books": "整理图书", "help_reader": "帮助读者", "patrol_library": "巡查书库",
}

# =============================================================================
# 各角色行为→地点映射
# =============================================================================
ROLE_ACTION_LOCATION = {
    "student": {
        "go_class": ["teaching_1", "teaching_2"],
        "study": ["library", "dorm_zone", "dorm_zone"],  # 学生2/3概率在宿舍学习，1/3在图书馆
        "eat": ["cafeteria_1", "cafeteria_2"],
        "chat": None,  # None 表示原地
        "exercise": "sports_field",
        "rest": "dorm_zone",
        "wander": "main_road",
    },
    "teacher": {
        "teach_class": ["teaching_1", "teaching_2"],
        "prepare_lesson": "library",
        "grade": ["teaching_1", "teaching_2"],
        "office_hours": ["teaching_1", "teaching_2"],
        "meeting": ["teaching_1", "teaching_2"],
        "chat": None,
        "rest": "staff_dorm",
        "eat": ["cafeteria_1", "cafeteria_2"],
        "wander": "main_road",
    },
    "cafeteria_staff": {
        "cook": "cafeteria_1",
        "serve": "cafeteria_1",
        "clean_kitchen": "cafeteria_1",
        "chat": None,
        "rest": "staff_dorm",
        "eat": "cafeteria_1",
        "wander": "main_road",
    },
    "librarian": {
        "manage_books": "library",
        "help_reader": "library",
        "patrol_library": "library",
        "chat": None,
        "rest": "staff_dorm",
        "eat": ["cafeteria_1", "cafeteria_2"],
        "wander": "main_road",
    },
}

# =============================================================================
# 各角色行为对状态的影响 (energy, hunger, social, knowledge, mood)
# 基础值，实际值会乘以 TIME_SCALE
# =============================================================================
ROLE_ACTION_DELTAS = {
    "student": {
        "go_class":  {"energy": -0.08, "hunger": -0.06, "social": +0.02, "knowledge": +0.10, "mood": -0.02},
        "study":     {"energy": -0.10, "hunger": -0.05, "social": -0.03, "knowledge": +0.12, "mood": -0.03},
        "eat":       {"energy": +0.03, "hunger": +0.30, "social": +0.02, "knowledge":  0.00, "mood": +0.05},
        "chat":      {"energy": -0.03, "hunger": -0.02, "social": +0.20, "knowledge": +0.02, "mood": +0.10},
        "exercise":  {"energy": -0.15, "hunger": -0.10, "social": +0.05, "knowledge":  0.00, "mood": +0.12},
        "rest":      {"energy": +0.25, "hunger": -0.03, "social": -0.02, "knowledge":  0.00, "mood": +0.05},
        "wander":    {"energy": -0.04, "hunger": -0.03, "social": +0.03, "knowledge": +0.01, "mood": +0.03},
    },
    "teacher": {
        "teach_class":    {"energy": -0.12, "hunger": -0.06, "social": +0.08, "knowledge": +0.03, "mood": -0.03},
        "prepare_lesson": {"energy": -0.08, "hunger": -0.04, "social": -0.02, "knowledge": +0.06, "mood":  0.00},
        "grade":          {"energy": -0.10, "hunger": -0.04, "social": -0.03, "knowledge": +0.02, "mood": -0.05},
        "office_hours":   {"energy": -0.06, "hunger": -0.03, "social": +0.10, "knowledge": +0.03, "mood": +0.02},
        "meeting":        {"energy": -0.08, "hunger": -0.04, "social": +0.05, "knowledge": +0.02, "mood": -0.04},
        "eat":            {"energy": +0.03, "hunger": +0.30, "social": +0.02, "knowledge":  0.00, "mood": +0.05},
        "chat":           {"energy": -0.03, "hunger": -0.02, "social": +0.15, "knowledge": +0.02, "mood": +0.08},
        "rest":           {"energy": +0.20, "hunger": -0.03, "social": -0.02, "knowledge":  0.00, "mood": +0.05},
        "wander":         {"energy": -0.04, "hunger": -0.03, "social": +0.03, "knowledge": +0.01, "mood": +0.03},
    },
    "cafeteria_staff": {
        "cook":          {"energy": -0.15, "hunger": -0.08, "social": +0.02, "knowledge":  0.00, "mood":  0.00},
        "serve":         {"energy": -0.10, "hunger": -0.05, "social": +0.12, "knowledge":  0.00, "mood": +0.03},
        "clean_kitchen": {"energy": -0.12, "hunger": -0.06, "social":  0.00, "knowledge":  0.00, "mood": -0.03},
        "eat":           {"energy": +0.03, "hunger": +0.30, "social": +0.02, "knowledge":  0.00, "mood": +0.05},
        "chat":          {"energy": -0.03, "hunger": -0.02, "social": +0.15, "knowledge":  0.00, "mood": +0.08},
        "rest":          {"energy": +0.20, "hunger": -0.03, "social": -0.02, "knowledge":  0.00, "mood": +0.05},
        "wander":        {"energy": -0.04, "hunger": -0.03, "social": +0.03, "knowledge":  0.00, "mood": +0.03},
    },
    "librarian": {
        "manage_books":   {"energy": -0.08, "hunger": -0.04, "social": -0.02, "knowledge": +0.04, "mood":  0.00},
        "help_reader":    {"energy": -0.06, "hunger": -0.03, "social": +0.12, "knowledge": +0.03, "mood": +0.05},
        "patrol_library": {"energy": -0.05, "hunger": -0.03, "social": +0.02, "knowledge": +0.01, "mood": +0.02},
        "eat":            {"energy": +0.03, "hunger": +0.30, "social": +0.02, "knowledge":  0.00, "mood": +0.05},
        "chat":           {"energy": -0.03, "hunger": -0.02, "social": +0.15, "knowledge": +0.02, "mood": +0.08},
        "rest":           {"energy": +0.20, "hunger": -0.03, "social": -0.02, "knowledge":  0.00, "mood": +0.05},
        "wander":         {"energy": -0.04, "hunger": -0.03, "social": +0.03, "knowledge": +0.01, "mood": +0.03},
    },
}

# =============================================================================
# 各角色的 LLM 决策 SYSTEM_PROMPT
# =============================================================================
ROLE_SYSTEM_PROMPTS = {
    "student": (
        "你是一名大学生，需要决定下一步做什么。\n"
        "请根据提供的信息，做出符合你性格和目标的选择。\n"
        "【重要】：在description中只描述活动内容，不要包含地点信息。\n"
        "地点信息会由系统自动添加。\n"
        "例如：不要说'在图书馆自习'或'准备去图书馆学习'，而是说'专心复习高等数学'。\n"
        "请严格以JSON格式返回，格式为：\n"
        "{\n"
        '  "action": "行为名",\n'
        '  "reason": "一句话理由",\n'
        '  "description": "详细描述你现在正在做什么（只描述活动内容，不包含地点）",\n'
        '  "duration_ticks": 持续tick数（建议1-6之间的整数）,\n'
        '  "contextual_effects": {\n'
        '    "mood": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "hunger": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "energy": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "social": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "knowledge": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."}\n'
        '  }\n'
        "}\n"
        "只返回JSON，不要有任何其他内容。\n"
        "注意：contextual_effects是可选的，如果没有特殊情况可以不提供，或只提供有变化的状态。"
    ),
    "teacher": (
        "你是一名大学教师，需要决定下一步做什么。\n"
        "你的主要职责是教学、备课、批改作业和指导学生。\n"
        "请根据提供的信息，做出符合你职责和状态的选择。\n"
        "【重要】：在description中只描述活动内容，不要包含地点信息。\n"
        "请严格以JSON格式返回，格式为：\n"
        "{\n"
        '  "action": "行为名",\n'
        '  "reason": "一句话理由",\n'
        '  "description": "详细描述你现在正在做什么（只描述活动内容，不包含地点）",\n'
        '  "duration_ticks": 持续tick数（建议1-6之间的整数）,\n'
        '  "contextual_effects": {\n'
        '    "mood": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "hunger": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "energy": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "social": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "knowledge": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."}\n'
        '  }\n'
        "}\n"
        "只返回JSON，不要有任何其他内容。"
    ),
    "cafeteria_staff": (
        "你是一名高校食堂工作人员，需要决定下一步做什么。\n"
        "你的主要职责是做饭、打饭和清洁厨房，工作时间集中在饭点前后。\n"
        "请根据提供的信息，做出符合你工作安排和状态的选择。\n"
        "【重要】：在description中只描述活动内容，不要包含地点信息。\n"
        "请严格以JSON格式返回，格式为：\n"
        "{\n"
        '  "action": "行为名",\n'
        '  "reason": "一句话理由",\n'
        '  "description": "详细描述你现在正在做什么（只描述活动内容，不包含地点）",\n'
        '  "duration_ticks": 持续tick数（建议1-6之间的整数）,\n'
        '  "contextual_effects": {\n'
        '    "mood": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "hunger": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "energy": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "social": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "knowledge": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."}\n'
        '  }\n'
        "}\n"
        "只返回JSON，不要有任何其他内容。"
    ),
    "librarian": (
        "你是一名高校图书馆管理员，需要决定下一步做什么。\n"
        "你的主要职责是整理图书、帮助读者、巡查书库，工作时间是图书馆开放时段。\n"
        "请根据提供的信息，做出符合你工作安排和状态的选择。\n"
        "【重要】：在description中只描述活动内容，不要包含地点信息。\n"
        "请严格以JSON格式返回，格式为：\n"
        "{\n"
        '  "action": "行为名",\n'
        '  "reason": "一句话理由",\n'
        '  "description": "详细描述你现在正在做什么（只描述活动内容，不包含地点）",\n'
        '  "duration_ticks": 持续tick数（建议1-6之间的整数）,\n'
        '  "contextual_effects": {\n'
        '    "mood": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "hunger": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "energy": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "social": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."},\n'
        '    "knowledge": {"direction": "boost/normal/reduce", "magnitude": "small/medium/large", "reason": "..."}\n'
        '  }\n'
        "}\n"
        "只返回JSON，不要有任何其他内容。"
    ),
}

# =============================================================================
# 各角色的 LLM 反思 SYSTEM_PROMPT
# =============================================================================
ROLE_REFLECT_PROMPTS = {
    "student": """
你是一名大学生的"内心独白"。基于过去一天的行为记录，进行自我反思。

请从以下角度进行反思：
1. 今天的表现如何？符合自己的目标和性格吗？
2. 学习/社交/休息/运动是否平衡？
3. 哪些地方做得好？哪些地方需要改进？
4. 明天应该如何调整？

请以JSON格式返回：
{
    "reflection": "一段内心独白式的反思文字（100-200字）",
    "mood_adjustment": 情绪调整建议（-0.2 ~ +0.2之间的浮点数）,
    "motivation_adjustment": 动力调整建议（-0.2 ~ +0.2之间的浮点数）,
    "summary": "一句话总结"
}

只返回JSON，不要有其他内容。
""",
    "teacher": """
你是一名大学教师的"内心独白"。基于过去一天的工作记录，进行自我反思。

请从以下角度进行反思：
1. 今天的教学和科研工作完成情况如何？
2. 与学生的互动是否充分？有没有忽略学生的需求？
3. 备课和批改作业的进度是否合理？
4. 工作与休息的平衡如何？明天需要调整什么？

请以JSON格式返回：
{
    "reflection": "一段内心独白式的反思文字（100-200字）",
    "mood_adjustment": 情绪调整建议（-0.2 ~ +0.2之间的浮点数）,
    "motivation_adjustment": 动力调整建议（-0.2 ~ +0.2之间的浮点数）,
    "summary": "一句话总结"
}

只返回JSON，不要有其他内容。
""",
    "cafeteria_staff": """
你是一名高校食堂工作人员的"内心独白"。基于过去一天的工作记录，进行自我反思。

请从以下角度进行反思：
1. 今天的工作完成得怎么样？饭菜准备是否充分？
2. 与同学们的互动感受如何？
3. 体力消耗大不大？有没有好好休息？
4. 明天需要注意什么？

请以JSON格式返回：
{
    "reflection": "一段内心独白式的反思文字（100-200字）",
    "mood_adjustment": 情绪调整建议（-0.2 ~ +0.2之间的浮点数）,
    "motivation_adjustment": 动力调整建议（-0.2 ~ +0.2之间的浮点数）,
    "summary": "一句话总结"
}

只返回JSON，不要有其他内容。
""",
    "librarian": """
你是一名高校图书馆管理员的"内心独白"。基于过去一天的工作记录，进行自我反思。

请从以下角度进行反思：
1. 今天图书整理和读者服务做得怎么样？
2. 有没有帮到需要帮助的同学？
3. 工作节奏是否合理？有没有太累？
4. 明天有什么可以改进的？

请以JSON格式返回：
{
    "reflection": "一段内心独白式的反思文字（100-200字）",
    "mood_adjustment": 情绪调整建议（-0.2 ~ +0.2之间的浮点数）,
    "motivation_adjustment": 动力调整建议（-0.2 ~ +0.2之间的浮点数）,
    "summary": "一句话总结"
}

只返回JSON，不要有其他内容。
""",
}

# =============================================================================
# 各角色的工作时段（hour_in_day 值列表）
# 用于 Plan 决策中判断"是否在工作时间"
# =============================================================================
ROLE_WORK_HOURS = {
    "student": None,  # 学生用 class_group/schedule 决定
    "teacher": None,  # 老师用 teach_schedule 决定
    "cafeteria_staff": {
        # 食堂工作时段：早餐准备 6-9, 午餐准备 10-13, 晚餐准备 16-19
        "cook":  [0, 1, 3, 4, 5, 9, 10, 11],     # 饭点前做饭（hour_in_day）
        "serve": [2, 5, 6, 11, 12],                # 饭点打饭
        "clean_kitchen": [7, 13],                   # 饭后清洁
    },
    "librarian": {
        # 图书馆工作时段：8:00-21:00（hour_in_day 1-14）
        "manage_books":   [1, 2, 3, 8, 9, 10],
        "help_reader":    [4, 5, 6, 7, 11, 12, 13],
        "patrol_library": [3, 7, 10, 14],
    },
}

# =============================================================================
# 各角色的身份描述模板（用于 LLM prompt 中的身份介绍部分）
# =============================================================================
ROLE_IDENTITY_PREFIX = {
    "student": "大学生",
    "teacher": "大学教师",
    "cafeteria_staff": "食堂工作人员",
    "librarian": "图书馆管理员",
}
