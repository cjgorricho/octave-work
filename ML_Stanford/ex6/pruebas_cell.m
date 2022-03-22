
clear ; close all; home;

vec=['hugo ', 'paco ', 'luis ', 'hugo ', 'paco ', 'luis ', 'hugo ', 'paco ', 'luis ',...
     'hugo ', 'paco ', 'luis ', 'hugo ', 'paco ', 'luis ', 'hugo ', 'paco ', 'luis ',...
     'hugo ', 'paco ', 'luis ', 'hugo ', 'paco ', 'luis ', 'hugo ', 'paco ', 'luis ',...
     'hugo ', 'paco ', 'hugo ', 'paco ', 'hugo ', 'paco ', 'hugo ', 'hugo '];

c = {"",0};
ind = 0;
     
while ~isempty(vec)
  [str, vec] = strtok(vec);
  comp = max(strcmp(c, str));
  if (!comp)
    ind = ind + 1;
    c{ind, 1} = str;
    c{ind, 2} = 0;
    c{ind, 2} = c{ind, 2} + 1;
  else
    ind_word = find(strcmp(c, str));
    c{ind_word, 2} = c{ind_word, 2} + 1;
  endif

endwhile