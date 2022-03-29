function createVocabList(vocab_file, vocabList)

if (exist(vocab_file) != 0)
     vl_file = csv2cell(vocab_file);
     vocabList = [vl_file; vocabList];
     vec2sort = cell2mat(vocabList(:,2));
     [var1,idx] = sort(vec2sort, 'descend');
     vocabList = vocabList(idx,:);
     cell2csv(vocab_file, vocabList);      
else
     cell2csv(vocab_file, vocabList);
endif
    