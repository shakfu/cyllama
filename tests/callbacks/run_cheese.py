import cheese


def report_cheese(name):
    print("Found cheese: " + name)


cheese.find(report_cheese)


def my_callback(progress: float) -> int:
    print(f"my_callback progress: {progress * 10000}")
    return 0


def custom_callback(progress: float) -> int:
    print(f"custom_callback progress: {progress * 0.1}")
    return 0


params = cheese.CheeseParams()
params.call(10.2)
params.callback = my_callback
params.call(10.2)
params.call(10.2, callback=custom_callback)
