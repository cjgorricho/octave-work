

clf; clear;
fh = @(x1, x2) 6 - x2;
fplot (fh, [-10, 10]);
title ("fplot() sinc function (possible division by 0, near 0)");

