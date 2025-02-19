function [word_indices, vocabList] = processEmail_headers(email_contents)
%PROCESSEMAIL preprocesses a the body of an email and
%returns a list of word_indices 
%   word_indices = PROCESSEMAIL(email_contents) preprocesses 
%   the body of an email and returns a list of indices of the 
%   words contained in the email. 
%

% Load Vocabulary
% vocabList = getVocabList();

% Init values
word_indices = [];
vocabList = {"",0};
ind = 0;

% ========================== Preprocess Email ===========================

% Find the Headers ( \n\n and remove )
% Uncomment the following lines if you are working with raw emails with the
% full headers

bodystart = find(diff(strfind(email_contents, char(10))')<=2);
email_contents = email_contents(bodystart(1):end);

% Lower case
email_contents = lower(email_contents);

% Strip all HTML
% Looks for any expression that starts with < and ends with > and replace
% and does not have any < or > in the tag it with a space
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

% Handle Numbers
% Look for one or more characters between 0-9
email_contents = regexprep(email_contents, '[0-9]+', 'number');

% Handle URLS
% Look for strings starting with http:// or https://
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');

% Handle Email Addresses
% Look for strings with @ in the middle
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

% Handle $ sign
email_contents = regexprep(email_contents, '[$]+', 'dollar');


% ========================== Tokenize Email ===========================

% Output the email to screen as well
% fprintf('\n==== Processed Email ====\n\n');

% Process file
l = 0;

while ~isempty(email_contents)

    % Tokenize and also get rid of any punctuation
    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
   
    % Remove any non alphanumeric characters
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word 
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try str = porterStemmer(strtrim(str)); 
    catch str = ''; continue;
    end;

    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end

    % Print to screen, ensuring that the output lines are not too long
    %if (l + length(str) + 1) > 78
    %    fprintf('\n');
    %    l = 0;
    %end
    %fprintf('%s ', str);
    %l = l + length(str) + 1;
    
    comp = strcmp(vocabList, str);
    if (!comp)
      ind = ind + 1;
      vocabList{ind, 1} = str;
      vocabList{ind, 2} = 0;
      vocabList{ind, 2} = vocabList{ind, 2} + 1;
    else
      ind_word = find(strcmp(vocabList, str));
      vocabList{ind_word, 2} = vocabList{ind_word, 2} + 1;
    endif
    

end




% Print footer
%fprintf('\n\n=========================\n');

end
