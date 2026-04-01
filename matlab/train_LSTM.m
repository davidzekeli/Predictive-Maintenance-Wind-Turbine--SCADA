%% ==============================================================
% Healthy vs Damaged Classification — Coursework-Compliant Script
% (.mat input • signalDatastore • trainnet • BiLSTM • confusion chart)
% ==============================================================

clear; clc; close all;
rng(42);  % reproducibility

% EDIT THIS PATH if your folder is different:
rootDir = "C:\Users\DELL\Downloads\Wind_Turbine_Data";  % contains Healthy\ and Damaged\

%% 1) Import .mat signals using a robust ReadFcn
%    - tries common variable names; else first 2-D numeric matrix
%    - ensures each signal is [time x channels] (rows=time)
sigds = signalDatastore(rootDir, ...
    "IncludeSubfolders", true, ...
    "FileExtensions", ".mat", ...
    "ReadFcn", @(p) loadMatSignal(p));   % helper at bottom

files = sigds.Files;
fprintf("Datastore files found: %d\n", numel(files));
assert(~isempty(files), "No .mat files found under: %s", rootDir);

%% 2) Labels aligned to datastore file order (prevents index mismatch)
names  = cellfun(@getParentFolder, files, "UniformOutput", false);
labels = categorical(names);
disp("Label summary:"); summary(labels)

%% 3) Read all signals into memory
sigdata = readall(sigds);
fprintf("Loaded %d signals into memory.\n", numel(sigdata));
assert(numel(sigdata) == numel(labels), "Signals/labels length mismatch.");

%% 4) Split into Train / Val / Test (80 / 10 / 10)
idx = splitlabels(labels, [0.8 0.1]);
trainidx = idx{1}; validx = idx{2}; testidx = idx{3};

N = numel(sigdata);
assert(all(trainidx<=N) && all(validx<=N) && all(testidx<=N), "Split indices out of range.");

% Optional leakage guard
assert(isempty(intersect(trainidx, validx)) && ...
       isempty(intersect(trainidx, testidx)) && ...
       isempty(intersect(validx,  testidx)), "Overlap across splits detected.");

traindata   = sigdata(trainidx);
trainlabels = labels(trainidx);
valdata     = sigdata(validx);
vallabels   = labels(validx);
testdata    = sigdata(testidx);
testlabels  = labels(testidx);

fprintf("Split -> Train=%d | Val=%d | Test=%d\n", ...
    numel(traindata), numel(valdata), numel(testdata));

%% 5) Pad/Truncate sequences to 100 time steps (rows=time) — matrix-safe
L = 100;
traindata = cellfun(@(X) padMatrix(X, L), traindata, 'UniformOutput', false);
valdata   = cellfun(@(X) padMatrix(X, L), valdata,   'UniformOutput', false);
testdata  = cellfun(@(X) padMatrix(X, L), testdata,  'UniformOutput', false);

[numTime, numChannels] = size(traindata{1});
numClasses = numel(categories(labels));
fprintf("Input size -> time=%d, channels=%d, classes=%d\n", numTime, numChannels, numClasses);

%% 6) Define network (trainnet requires NO output layers)
layers = [
    sequenceInputLayer(numChannels)
    bilstmLayer(64, "OutputMode", "last")  % or lstmLayer(128,"OutputMode","last")
    fullyConnectedLayer(numClasses)
];

%% 7) Training options
opts = trainingOptions("adam", ...
    "ValidationData", {valdata, vallabels}, ...
    "Metrics", "accuracy", ...
    "Plots", "training-progress", ...
    "MaxEpochs", 100, ...
    "MiniBatchSize", 16);

%% 8) Train
 net = trainnet(traindata, trainlabels, layers, "crossentropy", opts);

%% 9) Test accuracy (print robustly as percentage)
acc = testnet(net, testdata, testlabels, "accuracy");
accPct = acc; if accPct <= 1, accPct = accPct * 100; end
fprintf("\nTest Accuracy: %.2f%%n", accPct);

%% 10) Predict scores -> labels (robust orientation for scores2label)
scores = minibatchpredict(net, testdata);
cls    = categories(trainlabels);
numCls = numel(cls);

% Normalize type/shape
if iscell(scores)
    try
        scores = cell2mat(scores);
    catch
        scores = vertcat(scores{:});
    end
end
if isa(scores, "dlarray"), scores = extractdata(scores); end
if ~ismatrix(scores), scores = squeeze(scores); end

% Determine feature (class) dimension for scores2label
if size(scores,2) == numCls
    featureDim = 2;  % [samples x classes] -> OK
elseif size(scores,1) == numCls
    featureDim = 1;  % [classes x samples]
else
    % try transpose once
    if size(scores.',2) == numCls
        scores = scores.'; featureDim = 2;
    else
        error("Cannot infer class dimension from scores of size %s (numClasses=%d).", ...
              mat2str(size(scores)), numCls);
    end
end

pred = scores2label(scores, cls, featureDim);

% Confusion matrix
figure; confusionchart(testlabels, pred);
title("Healthy vs Damaged - Confusion Matrix");

%% 11) (Optional) Save artefacts for submission
% save('healthy_damaged_net.mat','net');
% saveas(gcf, 'confusion_matrix_test.png');
% T = table(string(testlabels), string(pred), 'VariableNames', {'TrueLabel','PredLabel'});
% writetable(T, 'test_predictions.csv');

%% ====================== Helpers (keep below) =====================

function X = loadMatSignal(p)
%LOADMATSIGNAL Load a signal matrix from a .mat file.
% - Accepts char or string path.
% - Tries common variable names; else returns first 2-D numeric matrix.
% - Ensures orientation [time x channels] (rows=time).

    if isa(p,'string'), p = char(p); end
    S = load(p);

    % Try a few common names first
    likely = {'S','signal','data','acc','X','Y'};
    for k = 1:numel(likely)
        if isfield(S, likely{k})
            v = S.(likely{k});
            if isnumeric(v) && ismatrix(v) && ~isscalar(v)
                if size(v,1) < size(v,2), v = v.'; end  % ensure rows=time
                X = v; 
                return
            end
        end
    end

    % Fallback: first 2-D numeric variable
    fn = fieldnames(S);
    for k = 1:numel(fn)
        v = S.(fn{k});
        if isnumeric(v) && ismatrix(v) && ~isscalar(v)
            if size(v,1) < size(v,2), v = v.'; end
            X = v; 
            return
        end
    end

    error("No usable 2-D numeric matrix found in MAT file: %s", p);
end

function parent = getParentFolder(p)
% Return immediate parent folder name for a file path (e.g., Healthy/Damaged).
    [parentDir, ~, ~] = fileparts(p);
    [~, parent, ~]    = fileparts(parentDir);
end

function Y = padMatrix(X, L)
% Pad or truncate the TIME dimension (rows) to length L. Keep channels.
    T = size(X,1);
    if T > L
        Y = X(1:L, :);
    elseif T < L
        Y = [X; zeros(L-T, size(X,2), "like", X)];
    else
        Y = X;
    end
end
