from regime_lib.cli import main
from regime_lib.core.registry import METHOD_REGISTRY


def test_list_methods_uses_registered_inventory(capsys):
    assert main(["--list-methods"]) == 0

    output = capsys.readouterr().out
    assert output.splitlines() == [
        "Méthodes disponibles :",
        *(f"  - {name}" for name in sorted(METHOD_REGISTRY)),
    ]
