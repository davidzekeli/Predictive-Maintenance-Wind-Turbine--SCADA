function Y = prepOneFast(X, Tfixed)
    X = toCxT(X);           
    X = keep8(X);           
    X(~isfinite(X)) = 0;
    X = fixLength(X, Tfixed);
    Y = zscoreChan(X);      
end
