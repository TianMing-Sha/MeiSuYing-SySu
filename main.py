import os, json; os.environ['no_proxy'] = '*' # 避免代理网络产生意外污染

help_menu_description = \
"""
</br></br>普通对话使用说明: 1. 输入问题; 2. 点击提交
</br></br>基础功能区使用说明: 1. 输入文本; 2. 点击任意基础功能区按钮
</br></br>函数插件区使用说明: 1. 输入路径/问题, 或者上传文件; 2. 点击任意函数插件区按钮
</br></br>虚空终端使用说明: 点击虚空终端, 然后根据提示输入指令, 再次点击虚空终端
</br></br>如何保存对话: 点击保存当前的对话按钮
</br></br>如何语音对话: 请阅读Wiki
</br></br>如何临时更换API_KEY: 在输入区输入临时API_KEY后提交（网页刷新后失效）"""

def enable_log(PATH_LOGGING):
    import logging
    admin_log_path = os.path.join(PATH_LOGGING, "admin")
    os.makedirs(admin_log_path, exist_ok=True)
    log_dir = os.path.join(admin_log_path, "chat_secrets.log")
    try:logging.basicConfig(filename=log_dir, level=logging.INFO, encoding="utf-8", format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    except:logging.basicConfig(filename=log_dir, level=logging.INFO,  format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # Disable logging output from the 'httpx' logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
    print(f"所有对话记录将自动保存在本地目录{log_dir}, 请注意自我隐私保护哦！")



def concat_texts(a, b):
    return a + ',' + b

def clear_dropdown():
    return [] , [] , []

def main():
    import gradio as gr
    if gr.__version__ not in ['3.32.9']:
        raise ModuleNotFoundError("使用项目内置Gradio获取最优体验! 请运行 `pip install -r requirements.txt` 指令安装内置Gradio及其他依赖, 详情信息见requirements.txt.")
    from request_llms.bridge_all import predict
    from toolbox import format_io, find_free_port, on_file_uploaded, on_report_generated, get_conf, ArgsGeneralWrapper, DummyWith
    # 建议您复制一个config_private.py放自己的秘密, 如API和代理网址
    proxies, WEB_PORT, LLM_MODEL, CONCURRENT_COUNT, AUTHENTICATION = get_conf('proxies', 'WEB_PORT', 'LLM_MODEL', 'CONCURRENT_COUNT', 'AUTHENTICATION')
    CHATBOT_HEIGHT, LAYOUT, AVAIL_LLM_MODELS, AUTO_CLEAR_TXT = get_conf('CHATBOT_HEIGHT', 'LAYOUT', 'AVAIL_LLM_MODELS', 'AUTO_CLEAR_TXT')
    ENABLE_AUDIO, AUTO_CLEAR_TXT, PATH_LOGGING, AVAIL_THEMES, THEME, ADD_WAIFU = get_conf('ENABLE_AUDIO', 'AUTO_CLEAR_TXT', 'PATH_LOGGING', 'AVAIL_THEMES', 'THEME', 'ADD_WAIFU')
    NUM_CUSTOM_BASIC_BTN, SSL_KEYFILE, SSL_CERTFILE = get_conf('NUM_CUSTOM_BASIC_BTN', 'SSL_KEYFILE', 'SSL_CERTFILE')
    DARK_MODE, INIT_SYS_PROMPT, ADD_WAIFU, TTS_TYPE = get_conf('DARK_MODE', 'INIT_SYS_PROMPT', 'ADD_WAIFU', 'TTS_TYPE')
    if LLM_MODEL not in AVAIL_LLM_MODELS: AVAIL_LLM_MODELS += [LLM_MODEL]

    # 如果WEB_PORT是-1, 则随机选取WEB端口
    PORT = find_free_port() if WEB_PORT <= 0 else WEB_PORT
    from check_proxy import get_current_version
    from themes.theme import adjust_theme, advanced_css, theme_declaration, js_code_clear,js_code_clear2, js_code_reset, js_code_show_or_hide, js_code_show_or_hide_group2
    from themes.theme import js_code_for_css_changing, js_code_for_toggle_darkmode, js_code_for_persistent_cookie_init
    from themes.theme import load_dynamic_theme, to_cookie_str, from_cookie_str, assign_user_uuid
    title_html = f"<h1 align=\"center\">梅宿莺</h1>{theme_declaration}"
    title_html = f"<h1 align='center'>梅宿莺</h1><p align='center'>Manufactured by Sun Yat-Sen University</p>{theme_declaration}"

    # 对话、日志记录
    enable_log(PATH_LOGGING)

    # 一些普通功能模块
    from core_functional import get_core_functions
    functional = get_core_functions()

    # 高级函数插件
    from crazy_functional import get_crazy_functions
    DEFAULT_FN_GROUPS = get_conf('DEFAULT_FN_GROUPS')
    plugins = get_crazy_functions()
    all_plugin_groups = list(set([g for _, plugin in plugins.items() for g in plugin['Group'].split('|')]))
    match_group = lambda tags, groups: any([g in groups for g in tags.split('|')])

    # 处理markdown文本格式的转变
    gr.Chatbot.postprocess = format_io

    # 做一些外观色彩上的调整
    set_theme = adjust_theme()

    # 代理与自动更新
    from check_proxy import check_proxy, auto_update, warm_up_modules
    proxy_info = check_proxy(proxies)

    
    gr_L1 = lambda: gr.Row().style()
    gr_L2 = lambda scale, elem_id: gr.Column(scale=scale, elem_id=elem_id, min_width=400)
    if LAYOUT == "TOP-DOWN":
        gr_L1 = lambda: DummyWith()
        gr_L2 = lambda scale, elem_id: gr.Row()
        CHATBOT_HEIGHT /= 2

    cancel_handles = []
    customize_btns = {}
    predefined_btns = {}
    from shared_utils.cookie_manager import make_cookie_cache, make_history_cache
    with gr.Blocks(title="绿慧智体", theme=set_theme, analytics_enabled=False, css=advanced_css) as app_block:
        gr.HTML(title_html)
        secret_css = gr.Textbox(visible=False, elem_id="secret_css")


        cookies, web_cookie_cache = make_cookie_cache() # 定义 后端state（cookies）、前端（web_cookie_cache）两兄弟
        with gr_L1():
            with gr_L2(scale=2, elem_id="gpt-chat"):
                chatbot = gr.Chatbot(label=f"梅宿莺", elem_id="gpt-chatbot")
                if LAYOUT == "TOP-DOWN":  chatbot.style(height=CHATBOT_HEIGHT)
                history, history_cache, history_cache_update = make_history_cache() # 定义 后端state（history）、前端（history_cache）、后端setter（history_cache_update）三兄弟


            with gr_L2(scale=1, elem_id="gpt-panel"):
                with gr.Accordion("输入区", open=True, elem_id="input-panel") as area_input_primary:
                    with gr.Row():
                        # ------------------------------------------------------------------------------
                        
                        txta = gr.Textbox(show_label=False, placeholder="Input question here.", elem_id='user_input_1' , visible = True).style(container=False)
                        txtb = gr.Textbox(show_label=False, placeholder="Input question here.", elem_id='user_input_2' , visible = True).style(container=False)
                        txt = gr.Textbox(show_label=False, placeholder="Input question here.", elem_id='user_input_sum' , visible = False).style(container=False)

                        txta.input(fn=concat_texts, inputs=[txta, txtb], outputs=txt)
                        txtb.input(fn=concat_texts, inputs=[txta, txtb], outputs=txt)



                    with gr.Row():
                        submitBtn = gr.Button("提交", elem_id="elem_submit", variant="primary" , visible = False)
                    with gr.Row():
                        resetBtn = gr.Button("重置", elem_id="elem_reset", variant="secondary"); resetBtn.style(size="sm")
                        stopBtn = gr.Button("停止", elem_id="elem_stop", variant="secondary"); stopBtn.style(size="sm")
                        clearBtn = gr.Button("清除", elem_id="elem_clear", variant="secondary", visible=False); clearBtn.style(size="sm")
                    if ENABLE_AUDIO:
                        with gr.Row():
                            audio_mic = gr.Audio(source="microphone", type="numpy", elem_id="elem_audio", streaming=True, show_label=False).style(container=False)
                    with gr.Row():
                        # status = gr.Markdown(f"Tip: 按Enter提交, 按Shift+Enter换行。当前模型: {LLM_MODEL} \n {proxy_info}", elem_id="state-panel")
                        status = gr.Markdown(f"Tip: 1.先将文件上传到下面的文件下载区 \n 2.再点击下面的ESG报告解读")
                with gr.Accordion("基础功能区", open=True, elem_id="basic-panel" , visible=False) as area_basic_fn:
                    # with gr.Row():
                    for k in range(NUM_CUSTOM_BASIC_BTN):
                        customize_btn = gr.Button("自定义按钮" + str(k+1), visible=False, variant="secondary", info_str=f'基础功能区: 自定义按钮')
                        customize_btn.style(size="sm")
                        customize_btns.update({"自定义按钮" + str(k+1): customize_btn})
                    for k in functional:
                        if ("Visible" in functional[k]) and (not functional[k]["Visible"]): continue
                        variant = functional[k]["Color"] if "Color" in functional[k] else "secondary"
                        functional[k]["Button"] = gr.Button(k, variant=variant , visible=False, info_str=f'基础功能区: {k}')
                        functional[k]["Button"].style(size="sm")
                        predefined_btns.update({k: functional[k]["Button"]})
                with gr.Accordion("", open=True, elem_id="plugin-panel") as area_crazy_fn:
                    # with gr.Row():
                        # gr.Markdown("插件可读取“输入区”文本/路径作为参数（上传文件自动修正路径）")
                    # with gr.Row(elem_id="input-plugin-group"):
                    #     plugin_group_sel = gr.Dropdown(choices=all_plugin_groups, label='', show_label=False, value=DEFAULT_FN_GROUPS,
                    #                                   multiselect=True, interactive=True, elem_classes='normal_mut_select').style(container=False)
                    with gr.Row():
                        for k, plugin in plugins.items():
                            if not plugin.get("AsButton", True): continue
                            visible = True if match_group(plugin['Group'], DEFAULT_FN_GROUPS) else False
                            variant = plugins[k]["Color"] if "Color" in plugin else "secondary"
                            info = plugins[k].get("Info", k)
                            plugin['Button'] = plugins[k]['Button'] = gr.Button(k, variant=variant,
                                visible=visible, info_str=f'函数插件区: {info}').style(size="sm")
                    with gr.Row():
                        with gr.Accordion("更多函数插件" , visible=False, open=True):
                            dropdown_fn_list = []
                            for k, plugin in plugins.items():
                                if not match_group(plugin['Group'], DEFAULT_FN_GROUPS): continue
                                if not plugin.get("AsButton", True): dropdown_fn_list.append(k)     # 排除已经是按钮的插件
                                elif plugin.get('AdvancedArgs', False): dropdown_fn_list.append(k)  # 对于需要高级参数的插件，亦在下拉菜单中显示
                            with gr.Row():
                                dropdown = gr.Dropdown(dropdown_fn_list, value=r"打开插件列表", label="", show_label=False).style(container=False)
                            with gr.Row():
                                plugin_advanced_arg = gr.Textbox(show_label=True, label="高级参数输入区", visible=False,
                                                                 placeholder="这里是特殊函数插件的高级参数输入区").style(container=False)
                            with gr.Row():
                                switchy_bt = gr.Button(r"请先从插件列表中选择", variant="secondary").style(size="sm")
                    with gr.Row():
                        with gr.Accordion("“文件上传区”。", open=True , visible=False) as area_file_up:
                            file_upload = gr.Files(label="请上传pdf文件", file_count="multiple", elem_id="elem_upload" , visible = False)

        with gr.Floating(init_x="0%", init_y="0%", visible=True, width=None, drag="forbidden", elem_id="tooltip"):
            with gr.Row():
                with gr.Tab("上传文件", elem_id="interact-panel"):
                    gr.Markdown("请上传本地文件/压缩包供“函数插件区”功能调用。请注意: 上传文件后会自动把输入区修改为相应路径。")
                    file_upload_2 = gr.Files(label="请上传pdf文件", file_count="multiple", elem_id="elem_upload_float" , visible = False)

                with gr.Tab("参数调整", elem_id="interact-panel"):
                    md_dropdown = gr.Dropdown(AVAIL_LLM_MODELS, value=LLM_MODEL, elem_id="elem_model_sel", label="更换LLM模型/请求源" , visible = True).style(container=False)
                    top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.01,interactive=True, label="Top-p (nucleus sampling)",)
                    temperature = gr.Slider(minimum=-0, maximum=2.0, value=1.0, step=0.01, interactive=True, label="Temperature", elem_id="elem_temperature")
                    max_length_sl = gr.Slider(minimum=256, maximum=1024*32, value=4096, step=128, interactive=True, label="Local LLM MaxLength",)
                    system_prompt = gr.Textbox(show_label=True, lines=2, placeholder=f"System Prompt", label="System prompt", value=INIT_SYS_PROMPT, elem_id="elem_prompt")
                    temperature.change(None, inputs=[temperature], outputs=None,
                        _js="""(temperature)=>gpt_academic_gradio_saveload("save", "elem_prompt", "js_temperature_cookie", temperature)""")
                    system_prompt.change(None, inputs=[system_prompt], outputs=None,
                        _js="""(system_prompt)=>gpt_academic_gradio_saveload("save", "elem_prompt", "js_system_prompt_cookie", system_prompt)""")
                    md_dropdown.change(None, inputs=[md_dropdown], outputs=None,
                        _js="""(md_dropdown)=>gpt_academic_gradio_saveload("save", "elem_model_sel", "js_md_dropdown_cookie", md_dropdown)""")

                with gr.Tab("界面外观", elem_id="interact-panel"):
                    theme_dropdown = gr.Dropdown(AVAIL_THEMES, value=THEME, label="更换UI主题").style(container=False)
                    checkboxes = gr.CheckboxGroup(["基础功能区", "函数插件区", "浮动输入区", "输入清除键", "插件参数区"], value=["基础功能区", "函数插件区"], label="显示/隐藏功能区", elem_id='cbs').style(container=False)
                    opt = ["自定义菜单"]
                    value=[]
                    if ADD_WAIFU: opt += ["添加Live2D形象"]; value += ["添加Live2D形象"]
                    checkboxes_2 = gr.CheckboxGroup(opt, value=value, label="显示/隐藏自定义菜单", elem_id='cbsc').style(container=False)
                    dark_mode_btn = gr.Button("切换界面明暗 ☀", variant="secondary").style(size="sm")
                    dark_mode_btn.click(None, None, None, _js=js_code_for_toggle_darkmode)
                with gr.Tab("帮助", elem_id="interact-panel"):
                    gr.Markdown(help_menu_description)

        with gr.Floating(init_x="20%", init_y="50%", visible=False, width="40%", drag="top") as area_input_secondary:
            with gr.Accordion("浮动输入区", open=True, elem_id="input-panel2"):
                with gr.Row() as row:
                    row.style(equal_height=True)
                    with gr.Column(scale=10):
                        txt2 = gr.Textbox(show_label=False, placeholder="Input question here.",
                                          elem_id='user_input_float', lines=8, label="输入区2").style(container=False)
                    with gr.Column(scale=1, min_width=40):
                        submitBtn2 = gr.Button("提交", variant="primary"); submitBtn2.style(size="sm")
                        resetBtn2 = gr.Button("重置", variant="secondary"); resetBtn2.style(size="sm")
                        stopBtn2 = gr.Button("停止", variant="secondary"); stopBtn2.style(size="sm")
                        clearBtn2 = gr.Button("清除", elem_id="elem_clear2", variant="secondary", visible=False); clearBtn2.style(size="sm")


        with gr.Floating(init_x="20%", init_y="50%", visible=False, width="40%", drag="top") as area_customize:
            with gr.Accordion("自定义菜单", open=True, elem_id="edit-panel"):
                with gr.Row() as row:
                    with gr.Column(scale=10):
                        AVAIL_BTN = [btn for btn in customize_btns.keys()] + [k for k in functional]
                        basic_btn_dropdown = gr.Dropdown(AVAIL_BTN, value="自定义按钮1", label="选择一个需要自定义基础功能区按钮").style(container=False)
                        basic_fn_title = gr.Textbox(show_label=False, placeholder="输入新按钮名称", lines=1).style(container=False)
                        basic_fn_prefix = gr.Textbox(show_label=False, placeholder="输入新提示前缀", lines=4).style(container=False)
                        basic_fn_suffix = gr.Textbox(show_label=False, placeholder="输入新提示后缀", lines=4).style(container=False)
                    with gr.Column(scale=1, min_width=70):
                        basic_fn_confirm = gr.Button("确认并保存", variant="primary"); basic_fn_confirm.style(size="sm")
                        basic_fn_clean   = gr.Button("恢复默认", variant="primary"); basic_fn_clean.style(size="sm")

                        from shared_utils.cookie_manager import assign_btn__fn_builder
                        assign_btn = assign_btn__fn_builder(customize_btns, predefined_btns, cookies, web_cookie_cache)
                        # update btn
                        h = basic_fn_confirm.click(assign_btn, [web_cookie_cache, cookies, basic_btn_dropdown, basic_fn_title, basic_fn_prefix, basic_fn_suffix],
                                                   [web_cookie_cache, cookies, *customize_btns.values(), *predefined_btns.values()])
                        h.then(None, [web_cookie_cache], None, _js="""(web_cookie_cache)=>{setCookie("web_cookie_cache", web_cookie_cache, 365);}""")
                        # clean up btn
                        h2 = basic_fn_clean.click(assign_btn, [web_cookie_cache, cookies, basic_btn_dropdown, basic_fn_title, basic_fn_prefix, basic_fn_suffix, gr.State(True)],
                                                   [web_cookie_cache, cookies, *customize_btns.values(), *predefined_btns.values()])
                        h2.then(None, [web_cookie_cache], None, _js="""(web_cookie_cache)=>{setCookie("web_cookie_cache", web_cookie_cache, 365);}""")



        # 功能区显示开关与功能区的互动
        def fn_area_visibility(a):
            ret = {}
            ret.update({area_input_primary: gr.update(visible=("浮动输入区" not in a))})
            ret.update({area_input_secondary: gr.update(visible=("浮动输入区" in a))})
            ret.update({plugin_advanced_arg: gr.update(visible=("插件参数区" in a))})
            if "浮动输入区" in a: ret.update({txt: gr.update(value="")})
            return ret
        checkboxes.select(fn_area_visibility, [checkboxes], [area_basic_fn, area_crazy_fn, area_input_primary, area_input_secondary, txt, txt2, plugin_advanced_arg] )
        checkboxes.select(None, [checkboxes], None, _js=js_code_show_or_hide)

        # 功能区显示开关与功能区的互动
        def fn_area_visibility_2(a):
            ret = {}
            ret.update({area_customize: gr.update(visible=("自定义菜单" in a))})
            return ret
        checkboxes_2.select(fn_area_visibility_2, [checkboxes_2], [area_customize] )
        checkboxes_2.select(None, [checkboxes_2], None, _js=js_code_show_or_hide_group2)

        # 整理反复出现的控件句柄组合
        input_combo = [cookies, max_length_sl, md_dropdown, txt, txt2, top_p, temperature, chatbot, history, system_prompt, plugin_advanced_arg]
        output_combo = [cookies, chatbot, history, status]
        predict_args = dict(fn=ArgsGeneralWrapper(predict), inputs=[*input_combo, gr.State(True)], outputs=output_combo)
        # 提交按钮、重置按钮

        cancel_handles.append(txt.submit(**predict_args))
        cancel_handles.append(txt2.submit(**predict_args))
        cancel_handles.append(submitBtn.click(**predict_args))
        cancel_handles.append(submitBtn2.click(**predict_args))
        resetBtn.click(None, None, [chatbot, history, status], _js=js_code_reset)   # 先在前端快速清除chatbot&status
        resetBtn2.click(None, None, [chatbot, history, status], _js=js_code_reset)  # 先在前端快速清除chatbot&status
        reset_server_side_args = (lambda history: ([], [], "已重置", json.dumps(history)),
                                  [history], [chatbot, history, status, history_cache])
        resetBtn.click(*reset_server_side_args)    # 再在后端清除history，把history转存history_cache备用
        resetBtn2.click(*reset_server_side_args)   # 再在后端清除history，把history转存history_cache备用
        clearBtn.click(None, None, [txt, txt2], _js=js_code_clear2)
        clearBtn2.click(None, None, [txt, txt2], _js=js_code_clear2)
        if AUTO_CLEAR_TXT:
            submitBtn.click(None, None, [txt, txt2], _js=js_code_clear)
            submitBtn2.click(None, None, [txt, txt2], _js=js_code_clear)
            txt.submit(None, None, [txt, txt2], _js=js_code_clear)
            txt2.submit(None, None, [txt, txt2], _js=js_code_clear)
        # 基础功能区的回调函数注册
        for k in functional:
            if ("Visible" in functional[k]) and (not functional[k]["Visible"]): continue
            click_handle = functional[k]["Button"].click(fn=ArgsGeneralWrapper(predict), inputs=[*input_combo, gr.State(True), gr.State(k)], outputs=output_combo)
            cancel_handles.append(click_handle)
        for btn in customize_btns.values():
            click_handle = btn.click(fn=ArgsGeneralWrapper(predict), inputs=[*input_combo, gr.State(True), gr.State(btn.value)], outputs=output_combo)
            cancel_handles.append(click_handle)
        # 文件上传区，接收文件后与chatbot的互动
        file_upload.upload(on_file_uploaded, [file_upload, chatbot, txt, txt2, checkboxes, cookies], [chatbot, txt, txt2, cookies]).then(None, None, None,   _js=r"()=>{toast_push('上传完毕 ...'); cancel_loading_status();}")
        file_upload_2.upload(on_file_uploaded, [file_upload_2, chatbot, txt, txt2, checkboxes, cookies], [chatbot, txt, txt2, cookies]).then(None, None, None, _js=r"()=>{toast_push('上传完毕 ...'); cancel_loading_status();}")
        # 函数插件-固定按钮区
        for k in plugins:
            if not plugins[k].get("AsButton", True): continue
            click_handle = plugins[k]["Button"].click(ArgsGeneralWrapper(plugins[k]["Function"]), [*input_combo], output_combo)
            click_handle.then(on_report_generated, [cookies, file_upload, chatbot], [cookies, file_upload, chatbot]).then(None, [plugins[k]["Button"]], None, _js=r"(fn)=>on_plugin_exe_complete(fn)")
            cancel_handles.append(click_handle)
        # 函数插件-下拉菜单与随变按钮的互动
        def on_dropdown_changed(k):
            variant = plugins[k]["Color"] if "Color" in plugins[k] else "secondary"
            info = plugins[k].get("Info", k)
            ret = {switchy_bt: gr.update(value=k, variant=variant, info_str=f'函数插件区: {info}')}
            if plugins[k].get("AdvancedArgs", False): # 是否唤起高级插件参数区
                ret.update({plugin_advanced_arg: gr.update(visible=True,  label=f"插件[{k}]的高级参数说明：" + plugins[k].get("ArgsReminder", [f"没有提供高级参数功能说明"]))})
            else:
                ret.update({plugin_advanced_arg: gr.update(visible=False, label=f"插件[{k}]不需要高级参数。")})
            return ret
        dropdown.select(on_dropdown_changed, [dropdown], [switchy_bt, plugin_advanced_arg] )

        def on_md_dropdown_changed(k):
            return {chatbot: gr.update(label="当前模型："+k)}
        md_dropdown.select(on_md_dropdown_changed, [md_dropdown], [chatbot] )

        def on_theme_dropdown_changed(theme, secret_css):
            adjust_theme, css_part1, _, adjust_dynamic_theme = load_dynamic_theme(theme)
            if adjust_dynamic_theme:
                css_part2 = adjust_dynamic_theme._get_theme_css()
            else:
                css_part2 = adjust_theme()._get_theme_css()
            return css_part2 + css_part1

        theme_handle = theme_dropdown.select(on_theme_dropdown_changed, [theme_dropdown, secret_css], [secret_css])
        theme_handle.then(
            None,
            [secret_css],
            None,
            _js=js_code_for_css_changing
        )
        # 随变按钮的回调函数注册
        def route(request: gr.Request, k, *args, **kwargs):
            if k in [r"打开插件列表", r"请先从插件列表中选择"]: return
            yield from ArgsGeneralWrapper(plugins[k]["Function"])(request, *args, **kwargs)
        click_handle = switchy_bt.click(route,[switchy_bt, *input_combo], output_combo)
        click_handle.then(on_report_generated, [cookies, file_upload, chatbot], [cookies, file_upload, chatbot]).then(None, [switchy_bt], None, _js=r"(fn)=>on_plugin_exe_complete(fn)")
        cancel_handles.append(click_handle)
        # 终止按钮的回调函数注册
        stopBtn.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
        stopBtn2.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
        plugins_as_btn = {name:plugin for name, plugin in plugins.items() if plugin.get('Button', None)}
        def on_group_change(group_list):
            btn_list = []
            fns_list = []
            if not group_list: # 处理特殊情况：没有选择任何插件组
                return [*[plugin['Button'].update(visible=False) for _, plugin in plugins_as_btn.items()], gr.Dropdown.update(choices=[])]
            for k, plugin in plugins.items():
                if plugin.get("AsButton", True):
                    btn_list.append(plugin['Button'].update(visible=match_group(plugin['Group'], group_list))) # 刷新按钮
                    if plugin.get('AdvancedArgs', False): dropdown_fn_list.append(k) # 对于需要高级参数的插件，亦在下拉菜单中显示
                elif match_group(plugin['Group'], group_list): fns_list.append(k) # 刷新下拉列表
            return [*btn_list, gr.Dropdown.update(choices=fns_list)]
        # plugin_group_sel.select(fn=on_group_change, inputs=[plugin_group_sel], outputs=[*[plugin['Button'] for name, plugin in plugins_as_btn.items()], dropdown])

        app_block.load(assign_user_uuid, inputs=[cookies], outputs=[cookies])

        from shared_utils.cookie_manager import load_web_cookie_cache__fn_builder
        load_web_cookie_cache = load_web_cookie_cache__fn_builder(customize_btns, cookies, predefined_btns)
        app_block.load(load_web_cookie_cache, inputs = [web_cookie_cache, cookies],
            outputs = [web_cookie_cache, cookies, *customize_btns.values(), *predefined_btns.values()], _js=js_code_for_persistent_cookie_init)

        app_block.load(None, inputs=[], outputs=None, _js=f"""()=>GptAcademicJavaScriptInit("{DARK_MODE}","{INIT_SYS_PROMPT}","{ADD_WAIFU}","{LAYOUT}","{TTS_TYPE}")""")    # 配置暗色主题或亮色主题

    # gradio的inbrowser触发不太稳定，回滚代码到原始的浏览器打开函数
    def run_delayed_tasks():
        import threading, webbrowser, time
        print(f"如果浏览器没有自动打开，请复制并转到以下URL：")
        if DARK_MODE:   print(f"\t「暗色主题已启用（支持动态切换主题）」: http://localhost:{PORT}")
        else:           print(f"\t「亮色主题已启用（支持动态切换主题）」: http://localhost:{PORT}")

        def auto_updates(): time.sleep(0); auto_update()
        def open_browser(): time.sleep(2); webbrowser.open_new_tab(f"http://localhost:{PORT}")
        def warm_up_mods(): time.sleep(6);

        # threading.Thread(target=auto_updates, name="self-upgrade", daemon=True).start() # 查看自动更新
        threading.Thread(target=open_browser, name="open-browser", daemon=True).start() # 打开浏览器页面
        threading.Thread(target=warm_up_mods, name="warm-up",      daemon=True).start() # 预热tiktoken模块

    # 运行一些异步任务：自动更新、打开浏览器页面、预热tiktoken模块
    run_delayed_tasks()

    # 最后，正式开始服务
    from shared_utils.fastapi_server import start_app
    start_app(app_block, CONCURRENT_COUNT, AUTHENTICATION, PORT, SSL_KEYFILE, SSL_CERTFILE)


if __name__ == "__main__":
    main()
