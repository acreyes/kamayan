from pathlib import Path

import kamayan.RuntimeParameters as RuntimeParameters
import kamayan.pyKamayan as pyKamayan


def get_test_file():
    return Path(".temp_inputs.in")


def get_test_block():
    block = "parthenon/mesh"
    parms = {
        "refinement": "adaptive",
        "numlevel": 3,
        "nx1": 128,
        "x1min": -0.5,
        "x1max": 0.5,
        "ix1_bc": "outflow",
        "ox1_bc": "outflow",
        "nx2": 128,
        "x2min": -0.5,
        "x2max": 0.5,
        "ix2_bc": "outflow",
        "ox2_bc": "outflow",
        "nx3": 1,
        "x3min": -0.5,
        "x3max": 0.5,
        "ix3_bc": "outflow",
        "ox3_bc": "outflow",
        "nghost": 4,
    }
    return block, parms


def get_parm(
    pin: pyKamayan.ParameterInput, block: str, key: str, value: int | str | bool | float
):
    """call the right pin.Get method."""
    if isinstance(value, bool):
        return pin.GetInt(block, key)
    elif isinstance(value, int):
        return pin.GetInt(block, key)
    elif isinstance(value, str):
        return pin.GetStr(block, key)
    elif isinstance(value, float):
        return pin.GetReal(block, key)

    raise KeyError("can't get parm")


def test_write_input():
    block, parms = get_test_block()
    parth_mesh = RuntimeParameters.RuntimeParametersBlock(block, parms)
    parth_job = RuntimeParameters.RuntimeParametersBlock(
        "parthenon/job", {"problem_id": "sedov"}
    )

    input_file = get_test_file()
    pin = RuntimeParameters.InputParameters(input_file)
    pin.add(parth_job)
    pin.add(parth_mesh)
    pin.write()

    # load from parthenon and check that we get the same values
    pman = pyKamayan.InitEnv(["test_program", "-i", str(input_file)])
    pin = pman.pinput
    pman.pinput.dump()

    nwrong = 0
    if pin.GetStr("parthenon/job", "problem_id") != "sedov":
        nwrong += 1
    for key, value in parms.items():
        try:
            if get_parm(pin, block, key, value) != value:
                nwrong += 1
        except KeyError:
            nwrong += 1

    pman.ParthenonFinalize()
    assert nwrong == 0


def test_write_block():
    block, parms = get_test_block()
    rp_block = RuntimeParameters.RuntimeParametersBlock(block, parms)
    nwrong = 0
    for key, value in parms.items():
        if value != rp_block[key]:
            nwrong += 1

    assert nwrong == 0, "RuntimeParametersBlock doesn't preserve the dict."

    test_file = get_test_file()
    with open(test_file, "w") as fid:
        rp_block.write_input(fid)

    with open(test_file, "r") as fid:
        header = fid.readline().strip("\n")
        assert header == f"<{block}>"
        nwrong = 0
        for line in fid:
            key, value = line.strip("\n").split("=")
            if value.strip() != str(parms[key.strip()]):
                nwrong += 1
        assert nwrong == 0


if __name__ == "__main__":
    test_write_block()
    test_write_input()
