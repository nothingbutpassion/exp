% -------------------------------------------------------------------------
% NOTES: 
% There are 4 coordinate frames:
% frame0: projectors-center frame (origin is ground projection point of the center)
% frame1: left projector frame
% frame2: right projector frame
% frame3: screen coordinate frame
% -------------------------------------------------------------------------
function main()
    IMG_PATH = "/home/yh/images/lenna.jpeg";     % image for testing

    PRJ_W = 1920;                                   % projector image width
    PRJ_H = 1080;                                   % projector image height
    FOV1_H = deg2rad(44.0);                         % left projector horizontal fov        
    FOV1_V = 2*atan(tan(0.5*FOV1_H)*PRJ_H/PRJ_W);   % left projector vertical fov
    FOV2_H = deg2rad(44.0);                         % right projector horizontal fov        
    FOV2_V = 2*atan(tan(0.5*FOV2_H)*PRJ_H/PRJ_W);   % right projector vertical fov
   
    D1 = 380;               % distance between 2 projectors
    D2 = 1200;              % distance from projectors-center to screen
    ALPHA = deg2rad(15);    % screen angle relative to projectors

    K1 = camera_matrix(PRJ_W, PRJ_H, FOV1_H, FOV1_V);
    K2 = camera_matrix(PRJ_W, PRJ_H, FOV2_H, FOV2_V);
    [T13, T23] = screen_projector_transforms(D1, D2, ALPHA);
    [H13, H23] = screen_projector_homographies(K1, K2, T13, T23);
    [H31, H32] = deal(inv(H13), inv(H23));
    [screen_r1, screen_r2] = get_screen_regions(H31, H32, PRJ_W, PRJ_H);
    [screen_r1, screen_r2, overlap_r] = rectify_screen_regions(screen_r1, screen_r2);
    buf_w = get_buffer_width(overlap_r, PRJ_W, PRJ_H);
    fprintf("C=%d, O=%d, R=%d\n", buf_w, 2*PRJ_W - buf_w, buf_w - PRJ_W);

    image = load_image(IMG_PATH, buf_w, PRJ_H);

    [buf1, buf2] = split_buffers(image, PRJ_W, PRJ_H);
    [Hl3, Hr3] = screen_buffer_homographies(overlap_r, PRJ_W, PRJ_H);
    Hl1 = Hl3 * H31;
    Hr2 = Hr3 * H32;

    img1 = warp_image(buf1, Hl1);
    img2 = warp_image(buf2, Hr2);
    imshow(img1);
    imshow(img2);
end

function img = load_image(image_file, target_w, target_h)
    src = imread(image_file);
    img = imresize(src, [target_h, target_w], "bicubic");
%     h = size(src, 1);
%     w = size(src, 2);
%     if w/h < target_w/target_h
%         tw = round(target_h*w/h);
%         th = target_h;
%     else
%         tw = target_w;
%         th = round(target_w*h/w);
%     end
%     x = round((target_w - tw)/2);
%     y = round((target_h - th)/2);
%     resized = imresize(src, [th, tw], "bicubic");
%     img = zeros(target_h, target_w, 3, 'uint8');
%     img(y+1:y+th, x+1:x+tw, :) = resized;
end

function K = camera_matrix(w, h, fov_h, fov_v)
    cx = 0.5*w;
    cy = 0.5*h;
    fx = 0.5*w/tan(0.5*fov_h);
    fy = 0.5*h/tan(0.5*fov_v);
    K = [fx, 0,  cx;
         0,  fy, cy;
         0,  0,  1 ];
end

function [T13, T23] = screen_projector_transforms(d1, d2, alpha)
    % frame0 (x0, y0, z0, 1) -> frame1 (x1, y1, z1, 1)
    T10 = [ 1,  0,  0,  d1/2;
            0,  0, -1,  0;
            0,  1,  0,  0;
            0,  0,  0,  1];
    % frame0 -> frame2
    T20 = [ 1,  0,  0, -d1/2;
            0,  0, -1,  0;
            0,  1,  0,  0;
            0,  0,  0,  1];
    % frame3 -> frame0
    T03 = [ cos(alpha), -sin(alpha), 0,     0;
            sin(alpha),  cos(alpha), 0,     d2;
            0,           0,          1,     0;
            0,           0,          0,     1];
    T13 = T10 * T03;     % fram3 -> frame 1
    T23 = T20 * T03;     % fram3 -> frame 2
end

function [H13, H23] = screen_projector_homographies(K1, K2, T13, T23)
    % screen point (x3, z3, 1) -> left projector image point (u1, v1, 1)
    H13 = zeros(3, 3);
    H13(:,1) = T13(1:3,1);    % 1st column of Rotation matrix
    H13(:,2) = T13(1:3,3);    % 3rd column of Roataion matrix
    H13(:,3) = T13(1:3,4);    % translation
    H13 = K1 * H13;
    % screen point (x3, z3, 1) -> left projector image point (u2, v2, 1)
    H23 = zeros(3, 3);
    H23(:,1) = T23(1:3,1);    % 1st column of Rotation matrix
    H23(:,2) = T23(1:3,3);    % 3rd column of Roataion matrix
    H23(:,3) = T23(1:3,4);    % translation
    H23 = K2 * H23;
end

function [screen_r1, screen_r2] = get_screen_regions(H31, H32, prj_w, prj_h)
    % the cornes of the projector, each line is a point
    corners = [ 0,      0; 
                prj_w,  0;
                prj_w,  prj_h; 
                0,      prj_h];
    screen_r1 = tranform_points(corners, H31);
    screen_r2 = tranform_points(corners, H32);
%     r1 = (H31 * corners')';     % (3x3 * 3x4)' -> 4*3
%     r1(:,1) = r1(:,1)./r1(:,3);
%     r1(:,2) = r1(:,2)./r1(:,3);
%     screen_r1 = r1(:,1:2);      % points in the left region
%     r2 = (H32 * corners')';     % (3x3 * 3x4)' -> 4*3
%     r2(:,1) = r2(:,1)./r2(:,3);
%     r2(:,2) = r2(:,2)./r2(:,3);
%     screen_r2 = r2(:,1:2);      % points in the right region
end

function [screen_r1, screen_r2, overlap_r] = rectify_screen_regions(r1, r2)
    [p1, p2, p3, p4] = deal(r1(1,:), r1(2,:), r1(3,:), r1(4,:));
    [p5, p6, p7, p8] = deal(r2(1,:), r2(2,:), r2(3,:), r2(4,:));
    % adjust top points
    [p1(2), p2(2), p5(2), p6(2)] = deal(min([p1(2), p2(2), p5(2), p6(2)])); 
    % adjust bottom points
    [p3(2), p4(2), p7(2), p8(2)] = deal(max([p3(2), p4(2), p7(2), p8(2)]));
    % adjust left region
    [p1(1), p4(1)] = deal(max([p1(1), p4(1)]));
    [p2(1), p3(1)] = deal(min([p2(1), p3(1)]));
    % adjust right region
    [p5(1), p8(1)] = deal(max([p5(1), p8(1)]));
    [p6(1), p7(1)] = deal(min([p6(1), p7(1)]));

    screen_r1 = [p1; p2; p3; p4];   % left region
    screen_r2 = [p5; p6; p7; p8];   % right region
    overlap_r = [p5; p2; p3; p8];   % overlap region
end


function buf_w = get_buffer_width(overlap, prj_w, prj_h)
    screen_w = overlap(3,1) - overlap(1,1);
    screen_h = abs(overlap(3,2) - overlap(1,2));
    overlap_w = round(prj_h*screen_w/screen_h);
    overlap_w = min(overlap_w, prj_w);
    buf_w = 2*prj_w - overlap_w;
end 

function [buf1, buf2] = split_buffers(image, prj_w, prj_h)
    buf_w = size(image, 2);
    x = buf_w - prj_w;      % (x, 0) is top left of right buffer
    ow = prj_w - x;         % overlap width

    % left buffer
    buf1 = zeros(prj_h, prj_w, 3);
    buf1(:,1:prj_w,:) = image(:,1:prj_w,:);
    buf1(:,x+1:x+ow,:) = buf1(:,x+1:x+ow,:).*smoothstep_mask(ow, prj_h, 1, 0);

    % right buffer
    buf2 = zeros(prj_h, prj_w, 3);
    buf2(:,1:prj_w,:) = image(:,x+1:x+prj_w,:);
    buf2(:,1:ow,:) = buf2(:,1:ow,:).*smoothstep_mask(ow, prj_h, 0, 1);
end


function [Hl3, Hr3] = screen_buffer_homographies(overlap_r, prj_w, prj_h)
    [p1, p2] = deal(overlap_r(1,:), overlap_r(3,:));
    h = abs(p2(2) - p1(2));  % p2(2) - p1(2) is negative
    sx = prj_h/h;            % sx is positive 
    sz = -prj_h/h;           % sz is negative
    p0 = [p2(1) - h*prj_w/prj_h, p1(2)];
    % homography: screen (x, z, 1) -> image (u, v, 1)
    Hl3 = [ sx,  0,  -sx*p0(1);
            0,  sz,  -sz*p0(2);
            0,   0,   1       ];
    Hr3 = [ sx,  0,   -sx*p1(1);
            0,   sz,  -sz*p1(2);
            0,   0,   1       ];
end

function dst_points = tranform_points(src_points, H)
    n = size(src_points, 1);
    d = (H*[src_points, ones(n, 1)]')';
    d(:,1) = d(:,1)./d(:,3);
    d(:,2) = d(:,2)./d(:,3);
    dst_points = d(:,1:2);
end

function mask = smoothstep_mask(w, h, s, e)
    t = linspace(s, e, w);
    t = 3*t.^2 - 2*t.^3;
    mask = repmat(reshape(t, [1, w, 1]), [h, 1, 3]);  % (H, W, 3)
end

function warped = warp_image(img, H)
    [h, w, c] = size(img);
    [X, Y] = meshgrid(1:w, 1:h);
    pts = [X(:)'; Y(:)'; ones(1, h*w)];
    
    src = H * pts;
    src = src ./ src(3, :);
    
    warped = zeros(h, w, c, 'uint8');
    for ch = 1:c
        warped(:,:,ch) = uint8(reshape(interp2(double(img(:,:,ch)), ...
            src(1,:), src(2,:), 'linear', 0), h, w));
    end
end


