function data = matlab_validate_active_observation(replayPath)
%MATLAB_VALIDATE_ACTIVE_OBSERVATION Validate the portable replay contract.
info = dir(replayPath);
assert(numel(info)==1 && ~info.isdir && info.bytes>0 && info.bytes<16*1024*1024, ...
    'ISAC:ReplayFile','Replay must be one regular JSON file smaller than 16 MiB.');
data = jsondecode(fileread(replayPath));
assert(data.schema_version==1 && strcmp(data.kind,'active-observation-study'), ...
    'ISAC:ReplaySchema','Unsupported replay schema.');
assert(strcmp(data.provenance.status,'completed'),'ISAC:Incomplete','Study did not complete.');
assert(isstruct(data.episodes) && ~isempty(data.episodes),'ISAC:NoEpisodes','No episodes.');
for i = 1:numel(data.episodes)
    episode = data.episodes(i);
    trace = episode.trace;
    assert(numel(trace)>=2,'ISAC:Trace','Insufficient timeline.');
    times = [trace.time_s];
    assert(all(isfinite(times)) && all(diff(times)>=0),'ISAC:Clock','Timeline is not monotone.');
    assert(all([trace.travelled_m] <= episode.config.travel_budget_m+1e-9), ...
        'ISAC:Budget','Travel cap exceeded.');
    assert(strcmp(trace(end).status,episode.metrics.final_status), ...
        'ISAC:Status','Final result and timeline disagree.');
    events = episode.measurements;
    for j = 1:numel(events)
        event = events(j);
        assert(event.decision_s<=event.capture_s && event.capture_s<=event.available_s, ...
            'ISAC:EventTime','Invalid decision/capture/availability order.');
        assert(numel(event.prediction_mean_deg)==2 && numel(event.prediction_sigma_deg)==2 ...
            && all(isfinite(event.prediction_mean_deg)) ...
            && all(isfinite(event.prediction_sigma_deg)) && all(event.prediction_sigma_deg>0), ...
            'ISAC:Prediction','Invalid predictive distribution.');
    end
    for j = 1:numel(trace)
        row = trace(j);
        assert(numel(row.weights)==2 && all(isfinite(row.weights)) ...
            && all(row.weights>=0) && abs(sum(row.weights)-1)<1e-8, ...
            'ISAC:Belief','Invalid evidence weights.');
        if ~strcmp(row.phase,'arrival')
            assert(isempty(row.observation_deg),'ISAC:Leak','Unreleased observation in timeline.');
            if j>1
                assert(max(abs(row.weights-trace(j-1).weights))<1e-8, ...
                    'ISAC:Leak','Belief changed without an arrival.');
            end
        else
            match = events([events.uid]==row.step+1);
            assert(numel(match)==1 && match.available_s<=row.time_s, ...
                'ISAC:Leak','Evidence shown before availability.');
        end
    end
end
end
