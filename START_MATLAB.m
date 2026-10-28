% Start the portable ISAC prediction/evidence replay. Base MATLAB only.
% Open this file on the new Mac and click Run. No Python runtime is required.
isacRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(isacRoot,'scripts'));
matlab_phy_smoke();
matlab_active_observation(fullfile(isacRoot,'examples','active_observation','matlab_replay.json'));
