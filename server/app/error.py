def error_message(
    schema_version: str,
    session_id: str,
    code: str,
    message: str,
    client_message_id: str | None = None,
    retryable: bool = False,
):
    payload = {
        "type": "error",
        "schema_version": schema_version,
        "session_id": session_id,
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if client_message_id is not None:
        payload["client_message_id"] = client_message_id
    return payload
