function report = matlab_active_observation_smoke(replayPath,outDir)
%MATLAB_ACTIVE_OBSERVATION_SMOKE Run on the new Mac with actual MATLAB.
% Exercises every recorded frame, exports an actual UI image and records version.
% A failure leaves its error evidence and is rethrown; it is never reported PASS.
assert(~isfolder(outDir) && ~isfile(outDir),'ISAC:Exists','Output must be a new directory.');
mkdir(outDir);
report = struct('status','running','matlab_version',version, ...
    'release',version('-release'),'platform',computer,'replay',char(replayPath));
try
    data = matlab_validate_active_observation(replayPath);
    fig = matlab_active_observation(replayPath);
    cleanup = onCleanup(@() delete(fig));
    total = 0;
    for i = 1:numel(data.episodes)
        fig.UserData.selectEpisode(i);
        for j = 1:numel(data.episodes(i).trace)
            fig.UserData.renderFrame(j);
            assert(fig.UserData.currentFrame==j,'ISAC:Frame','Viewer did not advance.');
            assert(strcmp(fig.UserData.currentStatus,data.episodes(i).trace(j).status), ...
                'ISAC:ViewStatus','Viewer status differs from evidence.');
            total = total+1;
        end
    end
    fig.UserData.selectEpisode(1);
    fig.UserData.renderFrame(numel(data.episodes(1).trace));
    exportapp(fig,fullfile(outDir,'matlab-replay.png'));
    report.status = 'passed';
    report.episodes = numel(data.episodes);
    report.frames = total;
    writelines(jsonencode(report,'PrettyPrint',true),fullfile(outDir,'matlab-report.json'));
    delete(cleanup);
catch error
    report.status = 'failed';
    report.identifier = error.identifier;
    report.message = error.message;
    writelines(jsonencode(report,'PrettyPrint',true),fullfile(outDir,'matlab-report.json'));
    rethrow(error);
end
end
