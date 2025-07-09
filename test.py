from trdg.generators import GeneratorFromDict

generator =  GeneratorFromDict(
    count=10,
    language='bn'
)

for img, lbl in generator:
    img.save(f'/app/out/{lbl}.png')