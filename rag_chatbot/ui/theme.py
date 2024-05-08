JS_LIGHT_THEME = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

CSS = """
.btn {
    background-color: #64748B;
    color: #FFFFFF;
    }

.stop_btn {
    background-color: #ff7373;
    color: #FFFFFF;
    }
"""
