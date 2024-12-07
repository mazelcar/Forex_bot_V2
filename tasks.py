from invoke import task

@task
def format(c):
    """Run automatic code formatters.

    Args:
        c: The invoke context object.
    """
    print("=== Starting Format Task ===")

    print("\n1. Running docformatter...")
    try:
        c.run("docformatter --in-place --recursive .")
    except Exception as e:
        print(f"docformatter error: {e}")

    print("=== Format Task Complete ===")

@task
def lint(c):
    """Run code style checking tools.

    Args:
        c: The invoke context object.
    """
    print("=== Starting Lint Task ===")

    print("\n1. Running pydocstyle...")
    try:
        c.run("pydocstyle .")
    except Exception as e:
        print(f"pydocstyle error: {e}")

    print("\n2. Running pylint...")
    try:
        c.run("pylint .")
    except Exception as e:
        print(f"pylint error: {e}")

    print("\n=== Lint Task Complete ===")

@task(format, lint)
def all(c):
    """Run all tasks in sequence.

    Args:
        c: The invoke context object.
    """
    pass