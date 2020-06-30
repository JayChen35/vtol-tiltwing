This folder contains statically linked (precompiled) libraries, also known as SLLs. 

[This](https://stackoverflow.com/questions/5022374/include-directories-vs-lib-directory-concept-question#:~:text=Essentially%2C%20you%20can%20look%20at,and%20what%20their%20description%20is.)
question from StackOverflow explains, on a basic level, the difference between the "include" and "lib" directories.

"If your functions don't change at all, surely it's better to compile them once and just insert the pre-compiled 
definitions into your executable whenever you recompile main.cpp? That is exactly the purpose of .lib files. They 
contain the pre-compiled code for definitions of the relevant functions. You still need the include file to let you 
know what functions exist in the lib."

[Here](https://www.youtube.com/watch?v=or1dAmUO8k0) is a great video explaining how to create and use SLLs.
