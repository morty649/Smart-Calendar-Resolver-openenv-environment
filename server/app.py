from calender_en.server.app import app, main as package_main


@app.get("/")
def root():
    return {"message": "Smart Calendar Resolver is running"}


def main() -> None:
    package_main()


if __name__ == "__main__":
    main()
