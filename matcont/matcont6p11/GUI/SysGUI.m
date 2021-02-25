classdef SysGUI
    
    methods(Static)
        
        function new
            global path_sys
            %path_sys = [pwd '/Systems/'];
            [path_sys, ~, ~] = fileparts(which('standard.m'));
            path_sys = [path_sys '/'];
            systems_standalone('init');
            systems_standalone();
        end
        function edit(systemname)
            
            if (~isstr(systemname))
                systemname = func2str(systemname);
            end
            
            warning off;
            global path_sys;
            [path_sys, ~, ~] = fileparts(which('standard.m'));
            path_sys = [path_sys '/'];
            global gds;
            load( [path_sys  systemname '.mat' ]);
            
            systems_standalone();
            warning on;
            
        end
        
        function userfunctions(systemname)
            
            if (~ischar(systemname))
                systemname = func2str(systemname);
            end
            global path_sys
            global gds;
            [path_sys, ~, ~] = fileparts(which('standard.m'));
            path_sys = [path_sys '/'];
            systemmatfile = [path_sys  systemname '.mat' ];
            load(systemmatfile);
            
            
            handle = userfun_standalone();
            if (ishandle(handle)) , uiwait(handle); end; %if 'userfun' not closed yet, stall..
            
            %save(systemmatfile , 'gds');  %nodig?
            %}
            
        end
        
    end

end