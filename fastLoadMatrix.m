function mat = fastLoadMatrix(p, likelyVars)
    S = load(p);
    for k = 1:numel(likelyVars)
        vn = likelyVars(k);
        if isfield(S, vn) && isnumeric(S.(vn)) && ~isscalar(S.(vn)) && ndims(S.(vn)) == 2
            mat = S.(vn);
            return
        end
    end
    fn = fieldnames(S);
    for k = 1:numel(fn)
        v = S.(fn{k});
        if isnumeric(v) && ~isscalar(v) && ndims(v) == 2
            mat = v; return;
        end
    end
    error('No numeric 2D matrix found in file: %s', p);
end
