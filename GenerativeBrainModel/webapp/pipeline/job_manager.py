import threading
import uuid

# Global job registry
tasks = {}


def submit(fn, *args, **kwargs):
    """
    Submit a function to run in background. Returns a job_id.
    """
    job_id = uuid.uuid4().hex
    # Initialize task entry
    tasks[job_id] = {"status": "pending", "result": None, "error": None}

    def _run():
        tasks[job_id]["status"] = "running"
        try:
            result = fn(*args, **kwargs)
            tasks[job_id]["result"] = result
            tasks[job_id]["status"] = "done"
        except Exception as e:
            import traceback
            tasks[job_id]["error"] = traceback.format_exc()
            tasks[job_id]["status"] = "error"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return job_id


def status(job_id):
    """
    Get the status (`pending`, `running`, `done`, `error`) and error if any.
    """
    task = tasks.get(job_id)
    if not task:
        return {"status": "not_found"}
    return {"status": task["status"], "error": task.get("error")}


def result(job_id):
    """
    Return the result object for a completed task, or None otherwise.
    """
    task = tasks.get(job_id)
    if task and task.get("status") == "done":
        return task.get("result")
    return None 