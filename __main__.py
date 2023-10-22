from sample import create_app, sample_app

app = create_app()
sample_app.setup(app)

app.run("localhost", 8080)