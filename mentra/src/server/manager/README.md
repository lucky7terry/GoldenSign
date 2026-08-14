# manager/

Per-user manager classes. Each manager handles one responsibility and is instantiated inside the `User` class (`session/User.ts`).

| Class                    | Responsibility                                      |
| ------------------------ | --------------------------------------------------- |
| `SessionManager`         | Thin lookup — `Map<userId, User>` with get/create/remove |
| `PhotoManager`           | Photo capture, in-memory storage, SSE broadcasting  |
| `TranscriptionManager`   | Speech-to-text listener, SSE broadcasting           |
| `AudioManager`           | Text-to-speech and audio stop                       |
| `InputManager`           | Button presses and touchpad gestures                |
| `WebRTCStreamManager`    | Managed WebRTC stream lifecycle and AI stream control messages |

Every manager (except `SessionManager`) receives a back-reference to its `User` so it can access `this.user.appSession` and `this.user.userId`.
