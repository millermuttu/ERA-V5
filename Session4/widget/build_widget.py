import json
import sys


def build(stats_path, template_path, out_path):
    try:
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"build_widget: cannot load stats json: {e}")

    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    if "__STATS__" not in template:
        sys.exit("build_widget: template missing __STATS__ placeholder")

    # Re-serialize and escape any </script> so the inline JSON can't break out.
    blob = json.dumps(stats, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__STATS__", blob)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    build("Session4/data/cleaned/stats.json",
          "Session4/widget/template.html",
          "Session4/widget/index.html")
    print("wrote Session4/widget/index.html")


if __name__ == "__main__":
    main()
