

def logging(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        with open("log.txt", "a") as f:
            print(f"{func.__name__} {args} {kwargs} {result}", file=f)
        return result
    return wrapper
