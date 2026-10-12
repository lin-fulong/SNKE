class StepPrinter:
    def __init__(self, title: str):
        self.title = title

    def phase(self, name: str, detail: str = ""):
        message = f"[{self.title}] {name}"
        if detail:
            message += f" | {detail}"
        print(message)

    def metric(self, name: str, value: str):
        print(f"[{self.title}] {name}: {value}")

    def item(self, message: str):
        print(f"[{self.title}] {message}")

    def ok(self, message: str, detail: str = ""):
        output = f"[{self.title}] {message}"
        if detail:
            output += f" | {detail}"
        print(output)


def iteration_status(name: str, step: int, total: int, **metrics):
    message = f"[{name}] step {step}/{total}"
    if metrics:
        formatted = []
        for key, value in metrics.items():
            formatted.append(f"{key}={value}")
        message += " | " + " ".join(formatted)
    print(message)
