function copy_or_link_file (input_filename, output_filename)
    if isunix(),
        command = sprintf('ln -s "$(readlink -f "%s")" "%s"', input_filename, output_filename);
        system(command);
    else
        copyfile(input_filename, output_filename);
    end
end