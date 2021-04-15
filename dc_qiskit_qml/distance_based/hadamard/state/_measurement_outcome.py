class MeasurementOutcome:

    label: int
    branch: int
    count: int

    def __init__(self, label, branch, count) -> None:
        self.count = count
        self.label = label
        self.branch = branch
