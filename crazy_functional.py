from toolbox import HotReload  # HotReload 的意思是热更新，修改函数插件后，不需要重启程序，代码直接生效
from toolbox import trimmed_format_exc


def get_crazy_functions():

    import sys
    sys.path.append('./crazy_functions')
    from crazy_functions.中文诗词翻译 import 中文诗词翻译
    from crazy_functions.相似度计算 import 相似度计算

    function_plugins = {
        "中文句子翻译为日语": {
            "Group": "学术",
            "Color": "stop",
            "AsButton": True,  # 加入下拉菜单中
            "Info": "中文诗词翻译,将其自动翻译为日本诗词",
            "Function": HotReload(中文诗词翻译),
        },
        "计算两个诗词之间的相似度": {
            "Group": "学术",
            "Color": "stop",
            "AsButton": True,  # 加入下拉菜单中
            "Info": "动态计算两个诗句之间的相似度",
            "Function": HotReload(相似度计算),
        },
    }


    """
    设置默认值:
    - 默认 Group = 对话
    - 默认 AsButton = True
    - 默认 AdvancedArgs = False
    - 默认 Color = secondary
    """
    for name, function_meta in function_plugins.items():
        if "Group" not in function_meta:
            function_plugins[name]["Group"] = "对话"
        if "AsButton" not in function_meta:
            function_plugins[name]["AsButton"] = True
        if "AdvancedArgs" not in function_meta:
            function_plugins[name]["AdvancedArgs"] = False
        if "Color" not in function_meta:
            function_plugins[name]["Color"] = "secondary"

    return function_plugins
