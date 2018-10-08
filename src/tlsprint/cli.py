import click


@click.group()
def main():
    pass


@main.command()
def learn():
    print('bla')
