# need to run follwing relavant to the current directory

-- python -m spacy init fill-config base_config.cfg config.cfg

# before running the following cammand we need to convert the pickel file to a spacy file. for that use preprocesspickeltospacy.ipynb python file . it will generate the correct spacy file
-- python -m spacy train ./config/config.cfg --output ./output --paths.train ./data/train.spacy --paths.dev ./data/test.spacy