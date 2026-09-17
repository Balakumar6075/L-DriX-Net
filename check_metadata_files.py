from pathlib import Path


SITE_PACKAGES = Path(
    r"C:\Users\Balakumar B\OneDrive\Desktop\L-DriX-Net\venv\Lib\site-packages"
)

print("=" * 70)
print("CHECKING PACKAGE METADATA FILES")
print("=" * 70)
print()

dist_info_folders = sorted(
    SITE_PACKAGES.glob("*.dist-info")
)

print("Found", len(dist_info_folders), ".dist-info folders")
print()

for folder in dist_info_folders:

    print(f"Checking: {folder.name}")

    metadata_file = folder / "METADATA"
    wheel_file = folder / "WHEEL"
    entry_points_file = folder / "entry_points.txt"

    for file in [
        metadata_file,
        wheel_file,
        entry_points_file,
    ]:

        if not file.exists():
            continue

        try:
            size = file.stat().st_size

            with open(
                file,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as f:
                content = f.read()

            print(
                f"  OK: {file.name} "
                f"({size} bytes)"
            )

        except Exception as error:

            print()
            print("  !!! PROBLEM !!!")
            print("  File:", file)
            print("  Error:", repr(error))
            print()

print()
print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)