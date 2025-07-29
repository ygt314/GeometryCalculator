import webview

from api_3d import api

window = webview.create_window('几何计算器', 'http://localhost:9000', js_api=api, maximized=True)
webview.start(debug=True)
