function addNoise(pathImages, extImages, noiseType, noiseLevel)
    %Aux variables
    pathRet = pwd();
    di = dir(pathImages);
    folders = {};
    aux = 1;

    %Get all subfolders
    for i=1:length(di)
        if (di(i).isdir == 1 && (strcmp(di(i).name,'.') == 0 && strcmp(di(i).name,'..') == 0))
            folders{aux} = di(i).name;
            aux = aux + 1;
        end
    end

    %Get new pah
    cd(pathImages);
    cd('..');
    newPath = pwd();
    newPath = strcat(newPath,'/',noiseType,'-',num2str(noiseLevel));

    %Check if exist
    if ~exist(newPath, 'dir')
        mkdir(newPath);
    end

    %Enter in new path
    cd(newPath);

    %Create subfolders and add noise
    for i=1:length(folders)
        newSubPath = strcat(newPath,'/',folders{i});

        %Check if exist
        if ~exist(newSubPath, 'dir')
            mkdir(newSubPath);
        end

        %Read original images
        images = dir(strcat(pathImages,'/',folders{i},'/','*.',extImages));
        imgVec = {};

        for j=1:length(images)
            imgVec{j} = imread(strcat(pathImages,'/',folders{i},'/',images(j).name));
        end

        %Add noise in images
        if strcmp(noiseType,'gaussian') == 1
            for j=1:length(imgVec)
                %Gaussian noise
                noise = imnoise(imgVec{j},'gaussian',0,noiseLevel/255);
                str = strcat(newSubPath,'/',images(j).name);
                disp(str)
                imwrite(noise,str);
            end
        elseif strcmp(noiseType,'poisson') == 1
            for j=1:length(imgVec)
                %Poisson noise
                %scale is a power of noise
                scale = power(10,noiseLevel);
                noise = scale * imnoise(im2double(imgVec{j})/scale, 'poisson');
                str = strcat(newSubPath,'/',images(j).name);
                disp(str)
                imwrite(noise,str);
            end
        elseif strcmp(noiseType,'sp') == 1
            for j=1:length(imgVec)
                %Salt and Pepper noise
                noise = imnoise(imgVec{j},'salt & pepper',noiseLevel)
                str = strcat(newSubPath,'/',images(j).name);
                disp(str)
                imwrite(noise,str);
            end
        end

        clear images imgVec

    end

    cd(pathRet);
end
