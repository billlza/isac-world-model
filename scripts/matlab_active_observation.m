function fig = matlab_active_observation(replayPath, varargin)
%MATLAB_ACTIVE_OBSERVATION Inspect recorded prediction/action/evidence events.
% Base MATLAB only. This viewer does not train or execute a neural network.
% MATLAB R2023b+ is the target; actual execution must be checked on the new Mac.
% Future measurements and evaluation-only truth are never drawn by the replay.

if nargin == 0 || isempty(replayPath)
    root = fileparts(fileparts(mfilename('fullpath')));
    replayPath = fullfile(root,'examples','active_observation','matlab_replay.json');
end
p = inputParser;
addParameter(p,'Visible','on',@(x) any(strcmp(x,{'on','off'})));
parse(p,varargin{:});
data = matlab_validate_active_observation(replayPath);
episodes = data.episodes;
names = arrayfun(@(x) sprintf('%s | %s | seed %d',x.scenario,x.method,x.seed), ...
    episodes,'UniformOutput',false);
currentEpisode = 1;
frameIndex = 1;
playing = false;
colors = [0.12 0.40 0.70; 0.85 0.35 0.13];

fig = uifigure('Name','ISAC: predict, observe, revise','Position',[80 80 1220 840], ...
    'Visible',p.Results.Visible);
layout = uigridlayout(fig,[6 4]);
layout.RowHeight = {34,40,'1x','1x',42,42};
layout.ColumnWidth = {'1x','1x','1x','1x'};
pick = uidropdown(layout,'Items',names,'ItemsData',1:numel(episodes), ...
    'ValueChangedFcn',@(src,~) selectEpisode(src.Value));
pick.Layout.Row = 1; pick.Layout.Column = [1 3];
compare = uibutton(layout,'Text','Compare all methods', ...
    'ButtonPushedFcn',@(~,~) compareMethods());
compare.Layout.Row = 1; compare.Layout.Column = 4;
slider = uislider(layout,'Limits',[1 numel(episodes(1).trace)], ...
    'Value',1,'MajorTicks',[],'MinorTicks',[], ...
    'ValueChangedFcn',@(~,event) renderFrame(round(event.Value)));
slider.Layout.Row = 2; slider.Layout.Column = [1 3];
play = uibutton(layout,'Text','Play / pause','ButtonPushedFcn',@(~,~) playback());
play.Layout.Row = 2; play.Layout.Column = 4;
sceneAx = uiaxes(layout); sceneAx.Layout.Row = 3; sceneAx.Layout.Column = [1 2];
predictionAx = uiaxes(layout); predictionAx.Layout.Row = 3; predictionAx.Layout.Column = [3 4];
beliefAx = uiaxes(layout); beliefAx.Layout.Row = 4; beliefAx.Layout.Column = [1 2];
costAx = uiaxes(layout); costAx.Layout.Row = 4; costAx.Layout.Column = [3 4];
status = uilabel(layout,'FontWeight','bold');
status.Layout.Row = 5; status.Layout.Column = [1 4];
scope = uilabel(layout,'Text', ...
    'Synthetic recorded experiment | Scores are uncalibrated | No material recognition or safety guarantee');
scope.Layout.Row = 6; scope.Layout.Column = [1 4];
% Public handles are used by the MATLAB smoke test to exercise every episode.
fig.UserData = struct('selectEpisode',@selectEpisode,'renderFrame',@renderFrame, ...
    'episodeCount',numel(episodes),'replayPath',char(replayPath));
renderFrame(1);

    function selectEpisode(index)
        validateattributes(index,{'numeric'},{'scalar','integer','>=',1,'<=',numel(episodes)});
        playing = false;
        currentEpisode = index;
        pick.Value = index;
        slider.Value = 1;
        slider.Limits = [1 numel(episodes(index).trace)];
        renderFrame(1);
    end

    function renderFrame(index)
        episode = episodes(currentEpisode);
        validateattributes(index,{'numeric'}, ...
            {'scalar','integer','>=',1,'<=',numel(episode.trace)});
        frameIndex = index;
        slider.Value = index;
        history = episode.trace(1:index);
        row = history(end);
        anchor = episode.anchor;
        cfg = episode.config;
        ray = [cosd(anchor.bearing_deg), sind(anchor.bearing_deg)];
        camera = row.lateral_m * [-ray(2), ray(1)];
        ranges = anchor.range_m + anchor.radial_speed_mps * row.time_s + [0 cfg.alternative_depth_m];
        cla(sceneAx); hold(sceneAx,'on');
        plot(sceneAx,0,0,'ks','MarkerFaceColor','k','DisplayName','RF origin');
        plot(sceneAx,camera(1),camera(2),'kd','MarkerFaceColor',[0.3 0.65 0.45], ...
            'DisplayName','Camera pose');
        for h = 1:2
            point = ranges(h)*ray;
            plot(sceneAx,point(1),point(2),'o','Color',colors(h,:), ...
                'MarkerFaceColor',colors(h,:),'DisplayName',data.hypotheses{h});
            plot(sceneAx,[camera(1) point(1)],[camera(2) point(2)],'--', ...
                'Color',colors(h,:),'HandleVisibility','off');
        end
        hold(sceneAx,'off'); grid(sceneAx,'on'); axis(sceneAx,'equal');
        xlabel(sceneAx,'x (m)'); ylabel(sceneAx,'y (m)');
        title(sceneAx,'Proposed worlds from RF evidence (truth hidden)');
        legend(sceneAx,'Location','best','Interpreter','none');

        cla(predictionAx); hold(predictionAx,'on');
        measurements = episode.measurements;
        for k = 1:numel(measurements)
            event = measurements(k);
            if event.decision_s > row.time_s
                continue;
            end
            for h = 1:2
                errorbar(predictionAx,event.uid + (h-1.5)*0.12, ...
                    event.prediction_mean_deg(h),2*event.prediction_sigma_deg(h), ...
                    'o','Color',colors(h,:),'LineWidth',1.2);
            end
            if event.available_s <= row.time_s && ~isempty(event.observation_deg)
                plot(predictionAx,event.uid,event.observation_deg,'kx','LineWidth',2,'MarkerSize',9);
            end
        end
        hold(predictionAx,'off'); grid(predictionAx,'on');
        xlim(predictionAx,[0.5 cfg.steps+0.5]);
        xlabel(predictionAx,'Measurement request'); ylabel(predictionAx,'Bearing (deg)');
        title(predictionAx,'Prediction +/-2 sigma; black x = released evidence');

        weights = zeros(numel(history),2);
        for q = 1:numel(history)
            weights(q,:) = history(q).weights(:).';
        end
        cla(beliefAx); hold(beliefAx,'on');
        for h = 1:2
            stairs(beliefAx,[history.time_s],weights(:,h),'Color',colors(h,:), ...
                'LineWidth',1.6,'DisplayName',data.hypotheses{h});
        end
        hold(beliefAx,'off'); ylim(beliefAx,[0 1]);
        xlim(beliefAx,[episode.trace(1).time_s episode.trace(end).time_s]);
        grid(beliefAx,'on'); xlabel(beliefAx,'Evidence availability clock (s)');
        ylabel(beliefAx,'Uncalibrated relative support');
        title(beliefAx,'Belief changes only after release');

        requests = sum(arrayfun(@(e) e.decision_s <= row.time_s,measurements));
        cla(costAx);
        bar(costAx,[row.travelled_m / cfg.travel_budget_m, requests / cfg.steps]);
        costAx.XTick = [1 2]; costAx.XTickLabel = {'Travel budget','Camera packet budget'};
        ylim(costAx,[0 1.05]); ylabel(costAx,'Fraction used'); grid(costAx,'on');
        title(costAx,sprintf('Reserved transport: %d bits (capacity proxy)',requests*cfg.packet_bits));
        status.Text = sprintf('t = %.3f s | %s | %s | received = %d | contradictions = %d', ...
            row.time_s,row.phase,row.status,row.accepted_observations,row.contradictions);
        fig.UserData.currentFrame = frameIndex;
        fig.UserData.currentStatus = row.status;
        fig.UserData.currentTime = row.time_s;
        drawnow;
    end

    function playback()
        if playing
            playing = false;
            return;
        end
        playing = true;
        while isvalid(fig) && playing && frameIndex < numel(episodes(currentEpisode).trace)
            renderFrame(frameIndex+1);
            pause(0.3);
        end
        playing = false;
    end

    function compareMethods()
        selected = data.summary(strcmp({data.summary.scenario},episodes(currentEpisode).scenario));
        comparison = uifigure('Name','Held-out outcomes: all methods','Position',[120 120 1120 500]);
        panel = uigridlayout(comparison,[1 2]);
        rates = uiaxes(panel);
        bar(rates,[[selected.ever_wrong_confirmation].' [selected.abstained].']);
        rates.XTick = 1:numel(selected); rates.XTickLabel = {selected.method};
        rates.TickLabelInterpreter = 'none'; rates.XTickLabelRotation = 25;
        ylim(rates,[0 1]); legend(rates,{'Wrong confirmation','Abstention'});
        title(rates,sprintf('%s: %d held-out episodes per method', ...
            episodes(currentEpisode).scenario,selected(1).episodes),'Interpreter','none');
        costs = uiaxes(panel);
        bar(costs,[selected.camera_transport_bits_reserved]/1024);
        costs.XTick = rates.XTick; costs.XTickLabel = rates.XTickLabel;
        costs.TickLabelInterpreter = 'none'; costs.XTickLabelRotation = 25;
        ylabel(costs,'Mean reserved transport (Kibit)');
        title(costs,'Compare achieved costs, not only common budget caps');
    end
end
