"""Pure-stdlib QA lifecycle and rejection fixtures; never imports Torch/CUDA."""
import argparse
import contextlib
import copy
import gc
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import weakref

import stage_checks_memory_safe as qa


def run():
    checks = []
    references, cleanup_counts, rows = [], [], []

    class Graph:
        pass

    def body():
        graph = Graph()
        references.append(weakref.ref(graph))
        loss = dict(owns_graph=graph)
        assert loss["owns_graph"] is graph
        return dict(synthetic_backward_finished=True)

    def cleanup():
        gc.collect()
        cleanup_counts.append(sum(reference() is not None for reference in references))
        assert cleanup_counts[-1] == 0, "A previous successful phase still owns a graph"

    with contextlib.redirect_stdout(io.StringIO()):
        for index in range(3):
            assert qa.isolated_phase(str(index), body, cleanup, rows) == dict(synthetic_backward_finished=True)
    assert cleanup_counts == [0,0,0] and all(row["status"] == "passed" for row in rows)
    checks.append("graph_owning_body_locals_dead_before_cleanup_and_next_phase")

    def expect_failure(body, expected, label):
        rows, cleaned = [], []
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                qa.isolated_phase(label, body, lambda: cleaned.append(True), rows)
            except expected:
                pass
            else:
                raise AssertionError("Synthetic failure was swallowed: " + label)
        assert cleaned == [True] and len(rows) == 1 and rows[0]["status"] == "failed"
        checks.append(label)

    expect_failure(lambda: (_ for _ in ()).throw(AssertionError("hard assertion")), AssertionError,
                   "assertion_failure_propagates_and_cleans_up")
    expect_failure(Graph, TypeError, "graph_return_rejected_instead_of_retained")
    expect_failure(lambda: {"unsafe": float("nan")}, ValueError, "nonfinite_summary_rejected")

    reference = dict(logits=[1.,2.], aux_logits=[[3.,4.],[5.,6.]])
    qa.assert_exact_parity(reference, copy.deepcopy(reference), copy.deepcopy(reference), lambda a,b:a == b)
    checks.append("exact_logits_and_two_DS_positive_fixture")
    for branch, key, label in (("fixed","logits","constant_logit_mismatch_rejected"),
                               ("alpha0","logits","alpha0_logit_mismatch_rejected"),
                               ("fixed","aux_logits","constant_DS_mismatch_rejected"),
                               ("alpha0","aux_logits","alpha0_DS_mismatch_rejected")):
        values = {"fixed":copy.deepcopy(reference), "alpha0":copy.deepcopy(reference)}
        if key == "logits":
            values[branch][key][1] += 1e-15
        else:
            values[branch][key][1][1] += 1e-14
        try:
            qa.assert_exact_parity(reference, values["fixed"], values["alpha0"], lambda a,b:a == b)
        except AssertionError:
            checks.append(label)
        else:
            raise AssertionError("Exact parity mismatch accepted")
    truncated = copy.deepcopy(reference); truncated["aux_logits"].pop()
    try: qa.assert_exact_parity(reference, truncated, copy.deepcopy(reference), lambda a,b:a == b)
    except AssertionError: checks.append("truncated_DS_zip_cannot_pass")
    else: raise AssertionError("Missing DS output accepted")
    try: qa.model_checks("cuda",128,1)
    except ValueError: checks.append("batch1_refused_before_Torch_import")
    else: raise AssertionError("Batch scope weakened")

    with tempfile.TemporaryDirectory(prefix="gta_qa_lifetime_") as temporary:
        root=Path(temporary)
        for folder in ("code","next_stage"):(root/folder).mkdir()
        source=root/"code/model.py";source.write_text("# synthetic original\n",encoding="utf-8")
        ignored=root/"next_stage/gate.json";ignored.write_text("{}",encoding="utf-8")
        before=qa.source_inventory(root)
        assert set(before)=={"code/model.py"}
        source.write_text("# synthetic modified\n",encoding="utf-8")
        assert qa.source_inventory(root)!=before
        checks.append("source_inventory_binds_changed_executable_bytes")
    command=[sys.executable,"-I","-S",str(Path(qa.__file__).resolve()),"--help"]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0 and "--batch-size {2}" in result.stdout
    checks.append("CLI_help_works_without_site_packages_Torch_or_CUDA")
    assert "torch" not in sys.modules
    return dict(status="passed",synthetic_guard_checks=len(checks),checks=checks,
                torch_imported=False,CUDA_executed=False,study_arrays_read=False,
                actual_model_QA_pass_claimed=False,
                limitations="Only stdlib lifetime/parity/rejection fixtures; real CUDA128 batch2 FP32+AMP remains unexecuted.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args=parser.parse_args();result=run()
    if args.output:
        with Path(args.output).open("x",encoding="utf-8") as stream:
            json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(result))
