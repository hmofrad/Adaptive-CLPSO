function [gbest_fit] = MiPSO(f,dim,nop)
    %Microscopic PSO (MiPSO)
    %=======================
    %==== initialising   ===
    %======================= 
%     clear,clc
%     f= 8;nop = 40;dim=30;
    %dim = 10; nop = 10; FE = 30,000;
    %dim = 30; nop = 40; FE = 200,000;
    %[gh fe]=ACPSO(13,30,30,3,0.01,0.01);
    endgen = 10^4;    % maximum Generation
    max_fe = 1*10^6; % current fitness evaluation
    global  orthm  action
    if f == 1;  Ub = 100;     end % Sphere
    if f == 2;  Ub = 2.048;   end % Rosenbrock
    if f == 3;  Ub = 32.768;  end % Ackley 3
    if f == 4;  Ub = 600;     end % Griewank 3
    if f == 5;  Ub = 0.5;     end % Weierstrass 3
    if f == 6;  Ub = 5.12;    end % Rastrigin   6
    if f == 7;  Ub = 5.12;    end % Rastrigin_noncont
    if f == 8;  Ub = 500;     end % Schewfel
    if f == 9;  Ub = 32.768;  end % Rotated Ackley
    if f == 10; Ub = 600;     end % Rotated Griewank
    if f == 11; Ub = 0.5;     end % Rotated Weierstrass
    if f == 12; Ub = 5.12;    end % Rotated Rastrigin
    if f == 13; Ub = 5.12;    end % Rotated Rastrigin_noncont
    if f == 14; Ub = 500;     end % Rotated Schewfel
    
    Lb = -Ub;    Vmin = -0.2*(Ub-Lb);	Vmax = -Vmin;
    spd=Vmin+2.*Vmax.*rand(nop,dim);  % initialize velocity
    x=Lb+(Ub-Lb).*rand(nop,dim);  % initialize position
    
    c = 1.49445;
    w_max = 0.9;  w_min = 0.4;
    orthm = orthm_generator(dim); % Rotation
    x(:,end+1)=fit_func({f},x); % calculate each particle fittness
    pbest = x; %initialize Best Particle Position
    [mn ind] = min(x(:,end));
    gbest = x(ind,:);
    i = 1; fe = 0;

    % initialise pbestf
    pbestf = []; pbestftmp = [];
    for i = 1:nop
        for j = 1:dim
            Pind = randi([1 nop]);  % Particle Index
            Dind = randi([1 dim]);  % Dimension Index
            pbestftmp = [pbestftmp  pbest(Pind,Dind)];
        end
        pbestf = [pbestf; pbestftmp];
        pbestftmp = [];
    end
    pbest_tmp = zeros(nop,1); % calculate pbest improvement
    m = ones(nop,1); % particles refreshing gap 
    
    % initilize automaton
    alpha=0.1;beta=0.1;
    act1 = 1; % action 1 : increament
    act2 = 2; % action 2 : decrement
    act3 = 3; % action 3 : do nothing
    action = [act1 act2 act3];
    r = size(action,2);
    p = repmat(1/r,nop,r);
    p_hist = [];    gbest_hist = [];    fe_hist = [];
    
     while fe<=max_fe
        w = w_max*((w_max-w_min)*i/endgen);
        for j=1:nop
            if mod(i,m(j)) == 0
                rw = randsrc(1000,1,[action; p(j,:)]); % roulette wheel
                act = rw(ceil(1000*rand));  % selected action
                switch (act)
                    case {1}    % increament m
                        if m(j)<20
                            m(j) = m(j) + 1;
                        end
                        pbestf(j,:) = exemplar(x,pbest,j);
                    case{2}     % decrement m 
                        if m(j) ~= 1
                            m(j) = m(j) - 1;
                        end
                        pbestf(j,:) = exemplar(x,pbest,j);
                    case{3}     % do nothing
                        pbestf(j,:) = exemplar(x,pbest,j);
                end
            end
            pbest_tmp(j) = pbest(j,end);
            if (x(j,end) < pbest(j,end))
                pbest(j,:) = x(j,:);
            end
            if (pbest(j,end) < gbest(end))
                gbest = pbest(j,:);
            end
        end
        spd = w.*spd+c.*rand(nop,dim).*(pbestf-x(:,1:end-1));
        spd=(spd>Vmax).*Vmax+(spd<=Vmax).*spd; 
        spd=(spd<(Vmin)).*(Vmin)+(spd>=(Vmin)).*spd;
        x(:,1:end-1) = x(:,1:end-1)+spd;
        x(:,end) = fit_func({f},x(:,1:end-1));
        for j=1:nop
            if pbest(j,end) < pbest_tmp(j)
               p(j,act) = p(j,act) + alpha.*(1 - p(j,act)); % desired action
               p(j,action ~=act) = (1-alpha)*p(j,action ~=act);
            else
                p(j,act) = (1 - beta).*p(j,act); % non-desired action
                p(j,action ~=act) = (beta/(r-1))+(1-beta).*p(j,action ~=act);            
            end
        end
        fe = fe + nop;
        i = i + 1;
        gbest_hist = [gbest_hist gbest(end)]; fe_hist = [fe_hist fe];
        if mod(i,500)==0,fprintf('fun=%u,Gene=%u,Fit_Eval=%u,Gbest=%e\n',f,i,fe,gbest(end)),end
        if ((i>1300 && gbest_hist(length(gbest_hist)-800) == gbest(end)) || gbest(end) ==0)
            break
        end
    end
    gbest_fit = gbest(end);
end
