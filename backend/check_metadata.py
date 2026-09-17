import importlib.metadata as metadata


print("=" * 60)
print("PYTHON PACKAGE METADATA DIAGNOSTIC")
print("=" * 60)

print()

count = 0

for dist in metadata.distributions():

    try:
        name = dist.metadata.get("Name", "UNKNOWN")
        version = dist.metadata.get("Version", "UNKNOWN")

        print(f"Checking: {name} {version}")

        # This is the operation Pydantic is performing
        entries = dist.entry_points

        print(f"  Entry points: {len(entries)}")

        count += 1

    except Exception as error:

        print()
        print("=" * 60)
        print("PROBLEM FOUND")
        print("=" * 60)

        print("Distribution:")
        print(dist)

        print()
        print("Error:")
        print(repr(error))

        print()
        print("Path:")

        try:
            print(dist._path)
        except Exception:
            print("Unable to determine path")

        print()
        print("=" * 60)

        break

else:

    print()
    print("=" * 60)
    print("NO PACKAGE METADATA ERROR FOUND")
    print("=" * 60)

print()
print("Packages checked:", count)