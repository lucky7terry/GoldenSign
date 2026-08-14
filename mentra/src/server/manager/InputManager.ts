import type { AppSession } from "@mentra/sdk";
import type { User } from "../session/User";

/**
 * InputManager — handles all physical input from the glasses (buttons + touchpad).
 *
 * Registers listeners on the AppSession and routes events to the
 * appropriate manager (e.g. single_tap → photo.takePhoto()).
 */
export class InputManager {
  constructor(private user: User) {}

  /** Wire up all button and touch listeners on the glasses session */
  setup(session: AppSession): void {
    this.setupButtons(session);
    this.setupTouch(session);
  }

  /** Button press handlers */
  private setupButtons(session: AppSession): void {
    session.events.onButtonPress(async (button) => {
      console.log(`[Button] ${this.user.userId}: ${button.buttonId} (${button.pressType})`);

      if (button.pressType === "long") {
        await this.user.webrtcStream.toggle();
        return;
      }

      // Quick press — take a photo
      await this.user.photo.takePhoto();
    });
  }

  /** Touchpad gesture handlers */
  private setupTouch(session: AppSession): void {
    session.events.onTouchEvent("single_tap", async () => {
      console.log(`[Touch] ${this.user.userId}: single_tap`);
      await this.user.photo.takePhoto();
    });
  }
}
