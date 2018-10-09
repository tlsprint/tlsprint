import click

from tlsprint.learn import learn_models


@click.group()
def main():
    pass


@main.command()
@click.argument('model_directory', type=click.Path(exists=True))
def learn(model_directory):
    models = learn_models(model_directory)
    print(f'Found {len(models)} unique implementations:')
    implementations = [sorted(servers) for servers in models.values()]
    print('\n\n'.join('\n'.join(servers) for servers in implementations))
