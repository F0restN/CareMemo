import site
import sys
from pathlib import Path

site_packages = site.getsitepackages()

custom_path = str(Path.resolve(Path.cwd() / "src"))

print("Desire code path: ", site_packages[0])

custom_site_package_file_path = Path (site_packages[0]) / "my_src_paths.pth"

print("site package file path: ", custom_site_package_file_path)


if custom_path in sys.path:
    print("Environment set up !!")
else:
    Path(custom_site_package_file_path).write_text(custom_path)
    print("Environment set up !")

for path in sys.path:
    print("Current site package paths: ", path)




