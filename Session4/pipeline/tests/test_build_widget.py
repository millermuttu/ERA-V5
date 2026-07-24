import json
import pytest
from build_widget import build


TEMPLATE = ('<!doctype html><html><body><h1>S4</h1>'
            '<script id="stats" type="application/json">__STATS__</script>'
            '<script>const S=JSON.parse(document.getElementById("stats").textContent);</script>'
            '</body></html>')


def test_injects_stats_and_strips_placeholder(tmp_path):
    stats = {"final": {"docs": 3, "tokens": 10}, "stages": []}
    sp = tmp_path / "stats.json"
    sp.write_text(json.dumps(stats), encoding="utf-8")
    tp = tmp_path / "template.html"
    tp.write_text(TEMPLATE, encoding="utf-8")
    out = tmp_path / "index.html"
    build(str(sp), str(tp), str(out))
    html = out.read_text(encoding="utf-8")
    assert "__STATS__" not in html
    assert '"tokens": 10' in html or '"tokens":10' in html
    # closing-script-safe: the injected JSON parses back cleanly
    body = html.split('type="application/json">')[1].split("</script>")[0]
    assert json.loads(body)["final"]["docs"] == 3


def test_malformed_stats_json_exits_nonzero(tmp_path):
    sp = tmp_path / "bad.json"
    sp.write_text("{not json", encoding="utf-8")
    tp = tmp_path / "template.html"
    tp.write_text(TEMPLATE, encoding="utf-8")
    with pytest.raises(SystemExit):
        build(str(sp), str(tp), str(tmp_path / "index.html"))
