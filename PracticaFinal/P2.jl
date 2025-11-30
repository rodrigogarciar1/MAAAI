using CSV
using DataFrames

function unifyDataset(folderName, fileName)
    fileData = DataFrame()
    for (path, _, files) in walkdir(folderName)
        for file in files
            fileData = vcat(fileData, CSV.read(joinpath(path, file), DataFrame), cols= :union)
        end
    end
    sort!(fileData)
    CSV.write(fileName, fileData)
end

function separateDataframe(df)
    return df[:,1], df[:,2:end-1], df[:,end]
end