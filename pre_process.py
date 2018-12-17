from utils.download import *
from utils.parser import *
from create_file import *
from create_label import *
params = torch.load('parameters.pth')
import os

print('Downloading and setting up requirements for sentiment classification...')

download_and_unzip(parama['base_path'], True, True, True, True)

create_file(params['base_path'])

lib_dir = os.path.join(params['base_path'], 'lib')
classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

command = 'javac -cp $%s lib/*.java' % classpath
os.system(command)

h = glob.glob(os.path.join(params['base_path'], '*/sentences.txt'))

#parsing is not used here
for file in h:
    dependency_parse(file, cp=classpath, tokenize=False)
    constituency_parse(file, cp=classpath, tokenize=False)


d = create_dictionary(os.path.join(params['base_path'], 'STB'))


create_label_file(os.path.join(params['base_path'], 'train'), d)
create_label_file(os.path.join(params['base_path'], 'test'), d)

print('Pre processing completed...')
