function [x, y] = devernay_edges(image, sigma, high_threshold, low_threshold)
    image = double(image'); % convert (possibly) uint8 to double    
    [X,Y] = size(image);
    image = image(:);
    
    if (sigma == 0)
        [Gx, Gy, modG] = compute_gradient(image,X,Y);
    else
        img_rect = reshape(image, [X,Y])';
        image_filtered = imgaussfilt(img_rect, sigma)';
        [Gx, Gy, modG] = compute_gradient(image_filtered(:), X, Y);
    end

    [Ex, Ey, modG, Gx, Gy] = compute_edge_points(modG, Gx, Gy, X,Y);
    [next, prev, Ex, Ey] = chain_edge_points(Ex, Ey, Gx, Gy, X,Y);
    [next, prev] = thresholds_with_hysteresis(next, prev, modG, X,Y, high_threshold, low_threshold);
    [x,y,~,~] = list_chained_edge_points(next, prev, Ex, Ey, X,Y);
end

function [Gx, Gy, modG] = compute_gradient(img, X,Y)
    img_vec = img;
    Gx   = [];
    Gy   = [];
    modG = [];
    
    for x = 2 : (X-2)
        for y = 2 : (Y-2)
            Gx(x+y*X) = double( img_vec((x+1)+(y*X)) - img_vec((x-1)+(y*X)) );
            Gy(x+y*X) = double( img_vec(x+((y+1)*X)) - img_vec(x+((y-1)*X)) );
            modG(x+y*X) = double( sqrt(Gx(x+(y*X)) * Gx(x+(y*X)) + Gy(x+(y*X)) * Gy(x+(y*X))) ); 
        end
    end
end

function [x, y, curve_limits, M] = list_chained_edge_points(next, prev, Ex, Ey, X, Y)
    N = 1;
    M = 1;
    x =[];
    y = [];

    for i = 1 : X*Y
       if prev(i) >= 0 || next(i) >= 0
           curve_limits(M) = int32(N);
           M = M + 1;
           k=i;
           n=prev(k);
           while (n>=0 && n ~= i)
              k = n;
              n = prev(k);
           end
           
           while 1
               x(N) = double(Ex(k));
               y(N) = double(Ey(k));
               N = N + 1;
               n= next(k);
               next(k) = -1;
               prev(k) = -1;               
               k = n;
               if k < 0  
                   break;
               end
           end
       end   
    end
    curve_limits(M) = int32(N);
end

function [next, prev, Ex, Ey] = chain_edge_points(Ex, Ey, Gx, Gy, X, Y)
    if isempty(Ex) || isempty(Ey) || isempty(Gx) || isempty(Gy) 
        error('chain_edge_points: invalid input');
        return;
    end
        
    next = -ones(X*Y, 1, 'int32');
    prev = -ones(X*Y, 1, 'int32');
    
    for x = 3 : (X-3)
        for y = 3 : (Y-3)
            if Ex(x+(y*X)) >= 0.0 && Ey(x+(y*X)) >= 0.0  
                from = int32(x + y * X);    
                fwd_s = double(0.0);         
                bck_s = double(0.0);        
                fwd = int32(-1);             
                bck = int32(-1);            
                
                for i = -2 : 2
                    for j = -2 : 2
                        to = int32(x + i + (y+j)*X);
                        s = double(chain(from, to, Ex, Ey, Gx, Gy, X, Y));
                        if s > fwd_s
                            fwd_s = s;  % set the new best forward chaining
                            fwd = to;
                        end
                        if s < bck_s
                            bck_s = s;  % set the new best backward chaining
                            bck = to;
                        end                      
                    end
                end   

                if fwd >= 0 && next(from) ~= fwd
                    alt = prev(fwd);
                    if alt < 0 || chain(alt, fwd, Ex, Ey, Gx, Gy, X, Y) < fwd_s
                        
                       if next(from) >= 0
                           prev(next(from)) = -1;
                       end
                       next(from) = fwd;

                       if alt >= 0
                           next(alt) = -1;
                       end
                       prev(fwd) = from;
                    end
                end
                
                if bck >= 0 && prev(from) ~= bck
                    alt = next(bck); 
                    if alt < 0 || chain(alt, bck, Ex,Ey,Gx,Gy, X,Y) > bck_s
                       if alt >= 0
                           prev(alt) = -1;
                       end
                       next(bck) = from;

                       if prev(from) >= 0
                           next(prev(from)) = -1;
                       end
                       prev(from) = bck;
                    end
                end
            
            end
        end
    end
end

function [next, prev] = thresholds_with_hysteresis(next, prev, modG, X,Y, th_h, th_l)
    if isempty(modG) || isempty(next) || isempty(prev)
        error('threshold_with_hysteresis: invalid input');
        return;
    end
    
    valid = false(X*Y, 1);
    for i = 1 : X*Y
       if (prev(i) >= 0 || next(i) >= 0) && ~valid(i) && modG(i) >= th_h
          valid(i) = true;
                 
          j=i;
          k = next(j);
          while j >= 0 && k >= 0 && ~valid(k)
              if modG(k) < th_l
                  next(j) = -1;
                  prev(k) = -1;
              else
                  valid(k) = true;
              end
              j = next(j);
              k = next(j);              
          end
          
          j=i;
          k = prev(j);
          while j >= 0 && k >= 0 && ~valid(k)              
              if modG(k) < th_l
                  prev(j) = -1;
                  prev(k) = -1;
              else 
                  valid(k) = true;
              end
              j = prev(j);
              k = prev(j);              
          end
       end
    end
    
    for i = 1 : X*Y
        if (prev(i) >= 0 || next(i) >= 0) && ~valid(i)
            prev(i) = -1;
            next(i) = -1;
        end
    end
end

function [Ex, Ey, modG, Gx, Gy] = compute_edge_points(modG, Gx, Gy, X, Y)
    if isempty(modG) || isempty(Gx) || isempty(Gy)
        error('compute_edge_points: invalid input');
        return;
    end
    
    Ex = -ones(X*Y, 1, 'double');
    Ey = -ones(X*Y, 1, 'double');
        
    for x = 3 : (X-3) 
        for y = 3 : (Y-3)
            Dx = 0;                               
            Dy = 0;                                
            mod = double( modG(x+(y*X)) );         
            L =   double( modG(x-1+(y*X)) );       
            R =   double( modG(x+1+(y*X)) );       
            U =   double( modG(x+((y+1)*X)) );     
            D =   double( modG(x+((y-1)*X)) );    
            gx =  double( abs(Gx(x+(y*X))) );      
            gy =  double( abs(Gy(x+(y*X))) );          
    
            if  greater(mod,L) && ~greater(R,mod) && gx >= gy
                Dx = 1;   
            elseif greater(mod,D) && ~greater(U,mod) && gx <= gy
                Dy = 1;  
            end

            if Dx > 0 || Dy > 0
               a = double( modG(x-Dx + ((y-Dy)*X)) );
               b = double( modG(x + (y * X)) );
               c = double( modG(x+Dx + ((y+Dy)*X)) );
               offset = double(0.5 * (a-c) / (a-b-b+c));

               %store edge point
               Ex(x+(y*X)) = double(x + (offset * Dx));
               Ey(x+(y*X)) = double(y + (offset * Dy));
            end
        end
    end
end

function out = chain(from, to, Ex, Ey, Gx, Gy, X, Y)
    if isempty(Gx) || isempty(Gy) || isempty(Ex) || isempty(Ey)
        error('chain: invalid input');
        return;
    end
        
    if from < 1 || to < 1 || from > (X*Y) || to > (X*Y)
        error('chain: one of the points is out of the image');
        return;
    end
    
    if from == to
        out = 0.0;
        return;
    end
    
    if Ex(from) < 0.0 || Ey(from) < 0.0 || Ex(to) < 0.0 || Ey(to) < 0.0
        out = 0.0;
        return;
    end
    
    dx = double(Ex(to) - Ex(from));
    dy = double(Ey(to) - Ey(from));
    if(Gy(from) * dx - Gx(from) * dy) * (Gy(to) * dx - Gx(to) * dy) <= 0.0
        out = 0.0;
        return; 
    end
    
    if (Gy(from) * dx - Gx(from) * dy) >= 0.0
        out =  double (1.0 / dist(Ex(from), Ey(from), Ex(to), Ey(to)));  % /*forward chaining  */
    else
        out = double(-1.0 / dist(Ex(from), Ey(from), Ex(to), Ey(to)));  % /* backward chaining */
    end
end

function out = dist(x1, y1, x2, y2)
    out = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
end

function out = greater(a, b)
    if a <= b
        out = false;
        return;
    end
    if (a-b) < eps 
        out = false;
        return;
    end
    out = true;
end