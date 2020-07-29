function saveCodeAsText(originalCodeDir, rootDir)

% Read original code
codeTxt = fileread(originalCodeDir);
[~, codeName, ext] = fileparts(originalCodeDir);

% Write code
codeName = [codeName ext];
newCodeDir = fullfile(rootDir, codeName);
fid = fopen(newCodeDir, 'w');
fprintf(fid, '%s', codeTxt);
fclose(fid);

end