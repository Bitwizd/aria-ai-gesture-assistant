import subprocess
import platform


class ARIACommandEngine:
    """
    ARIA's command/action layer.

    Converts recognized commands into safe computer actions.
    """

    def __init__(self):
        self.system = platform.system()

    def execute(self, command):
        """
        Execute a recognized ARIA command.
        """

        command = command.lower().strip()

        if command == "open safari":
            self.open_application("Safari")
            return "Opening Safari"

        elif command == "open vscode":
            self.open_application("Visual Studio Code")
            return "Opening Visual Studio Code"

        elif command == "open terminal":
            self.open_application("Terminal")
            return "Opening Terminal"

        elif command == "close safari":
            self.close_application("Safari")
            return "Closing Safari"

        elif command == "close vscode":
            self.close_application("Visual Studio Code")
            return "Closing Visual Studio Code"

        elif command == "close terminal":
            self.close_application("Terminal")
            return "Closing Terminal"

        elif command == "system info":
            return self.get_system_info()

        else:
            return f"I don't know that command yet: {command}"

    # ---------------------------------------------------------
    # APPLICATION CONTROL
    # ---------------------------------------------------------

    def open_application(self, app_name):

        if self.system == "Darwin":
            subprocess.Popen(["open", "-a", app_name])

        elif self.system == "Windows":
            subprocess.Popen(["start", app_name], shell=True)

        elif self.system == "Linux":
            subprocess.Popen([app_name.lower()])

    def close_application(self, app_name):

        if self.system == "Darwin":
            subprocess.run(
                ["osascript", "-e",
                 f'tell application "{app_name}" to quit'],
                check=False
            )

        elif self.system == "Windows":
            subprocess.run(
                ["taskkill", "/IM", f"{app_name}.exe", "/F"],
                check=False
            )

        elif self.system == "Linux":
            subprocess.run(
                ["pkill", "-f", app_name],
                check=False
            )

    # ---------------------------------------------------------
    # SYSTEM INFORMATION
    # ---------------------------------------------------------

    def get_system_info(self):

        return f"ARIA is running on {self.system}."


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    aria = ARIACommandEngine()

    print("ARIA Command Engine")
    print("-------------------")

    while True:

        command = input("ARIA > ")

        if command.lower() in ["exit", "quit"]:
            print("ARIA shutting down.")
            break

        response = aria.execute(command)

        print("ARIA:", response)
        