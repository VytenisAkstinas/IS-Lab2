clear all; close all; clc;
X=linspace(1,20,20); %20 skaitmenu iejimo vektorius
F=(1+0.6*sin(2*pi*X/0.7)+0.3*sin(2*pi*X))/2;
%Funkcija kurios elgesi imituos neuronu tinklas

figure(1)
plot(X,F,'b',X,F,'*'); %Rezultato, kurio tikimasi, grafikas
grid minor;
title('Tikrosios funkcijos grafikas');xlabel('Iejimo vektoriaus X verte'); ylabel('F(X)');

%% Pradinis tinklo ciklas

% Pradedame sukurdami atsitiktines reiksmes (kaip ir IS LAB1) 
%jungtims/neuronams ir bias'ams. 
%w1x, b1x - pirmo sluoksnio svoriai
%w2x, b2x - antro sluoksnio svoriai
w11 = randn;
w12 = randn;
w13 = randn;
w14 = randn;
b11 = randn;
b12 = randn;
b13 = randn;
w21 = randn;
w22 = randn;
w23 = randn;
w24 = randn;
b14 = randn;
b21 = randn;

%Susikuriame tuscias X*1 dydzio matricas paslepto sluoksnio perceptronu
%pasvertai sumai (H) ir ju isejimo vertems (y).
H11=zeros(length(X),1);
H12=zeros(length(X),1);
H13=zeros(length(X),1);
H14=zeros(length(X),1);
OUTH11=zeros(length(X),1);
OUTH12=zeros(length(X),1);
OUTH13=zeros(length(X),1);
OUTH14=zeros(length(X),1);

%Tokias pacias matricas sukuriam galutiniam tinklo isejimui ir paklaidoms
OUT = zeros(length(X),1);
e=zeros(length(X),1);


% Paskaiciuojame ka tik minetas matricas
for i = 1:20
    %H skaiciuojame su tokia pacia formule kaip IS LAB1
    H11(i)=X(i)*w11+b11;
    H12(i)=X(i)*w12+b12;
    H13(i)=X(i)*w13+b13;
    H14(i)=X(i)*w14+b14;
    %H perceptronu isejimus paskaiciuojame su sigmoidine aktyvavimo funkcija 
    OUTH11(i)=1/(1+exp(-H11(i)));
    OUTH12(i)=1/(1+exp(-H12(i)));
    OUTH13(i)=1/(1+exp(-H13(i)));
    OUTH14(i)=1/(1+exp(-H14(i)));
end

%Pradedame antra sluoksni 

for i = 1:20
    %Uzpildome isejimo verciu matrica pagal hidden sluoksnio isejima ir
    %atsitiktinus antro sluoksnio svorius
    OUT(i)=OUTH11(i)*w21+OUTH12(i)*w22+OUTH13(i)*w23+OUTH14(i)*w24+b21;
    %Sioje vietoje turetume OUT isistatyti i aktyvavimo funkcija
    %Kadangi ji turi buti tiesine ir yra laisvai pasirenkama, aktyvavimo
    %funkcija isivaizduosime kaip Y=a*X=1*X. Tiesines funkcijos koeficienta
    %parinkus vieneta, si zingsni realiai praleidziame. Taip viska
    %supaprastinsime skaiciuojant isvestines atnaujinant svorius.
    %Uzpildome paklaidos matrica atimdami surasta OUT is pradinio signalo F
    e(i)=F(i)-OUT(i);
end

%Inicijuojame tuscius kitu paklaidu vektorius
E=0;

%Randame visu santykiniu paklaidu suma
for i = 1:20
    E=E+(e(i)^2)/2;
end

figure(2)
hold on
plot(X,F,'rx',X,OUT,'go');
plot(X,F,'b',X,OUT,'k');
hold off
title('Signalas gautas isejimme po pirmo ciklo lyginant su originaliu');
legend('Originalus','OUT');

%% BACK PROPAGATION
n=0.1; %Mokymo zingsnis
MAXerror=0.1; %Maksimali leistina klaida

j=0;
%Back propagation algoritmu koreguosim vertes, kol bendra paklaida bus
%didesne uz maksimalia leistina
while (E>MAXerror) 
    j=j+1;
    
    for i= 1:20
        %Atnaujinami antro sluoksnio svoriai. Kadangi cia aktyvavimo
        %funkcija buvo labai paprasta, svoriai atnaujinami labai paprasta
        %formule:
        w21 = w21 + n*e(i)*OUTH11(i);
        w22 = w22 + n*e(i)*OUTH12(i);
        w23 = w23 + n*e(i)*OUTH13(i);
        w24 = w24 + n*e(i)*OUTH14(i);
        b21 = b21 + n*e(i);
        
        %Atnaujinami pirmo sluoksnio parametrai. Pirmame sluoksnyje buvo
        %naudojama netiesine aktyvavimo funkcija, todel parametru
        %atnaujinimo formule zymiai sudetingesne. Ji rasta pagal "Back
        %Propagation in Neural Network with an example" (Youtube).
        w11 = w11 + n*e(i)*w21*X(i)*exp(b11+w11*X(i))/((exp(b11+w11*X(i))+1)^2);
        w12 = w12 + n*e(i)*w22*X(i)*exp(b12+w12*X(i))/((exp(b12+w12*X(i))+1)^2);
        w13 = w13 + n*e(i)*w23*X(i)*exp(b13+w13*X(i))/((exp(b13+w13*X(i))+1)^2);
        w14 = w14 + n*e(i)*w24*X(i)*exp(b14+w14*X(i))/((exp(b14+w14*X(i))+1)^2);
        
        b11 = b11 + n*e(i)*w21*exp(b11+w11*X(i))/((exp(b11+w11*X(i))+1)^2);
        b12 = b12 + n*e(i)*w22*exp(b12+w12*X(i))/((exp(b12+w12*X(i))+1)^2);
        b13 = b13 + n*e(i)*w23*exp(b13+w13*X(i))/((exp(b13+w13*X(i))+1)^2);
        b14 = b14 + n*e(i)*w24*exp(b14+w14*X(i))/((exp(b14+w14*X(i))+1)^2);
        
       
    end

	% Su atnaujintais svoriais atliekame tuos pacius zingsnius kaip
	% anksciau ir randame nauja rezultata isejime.
    for i = 1:20
        H11(i)=X(i)*w11+b11;
        H12(i)=X(i)*w12+b12;
        H13(i)=X(i)*w13+b13;
        H14(i)=X(i)*w14+b14;
        OUTH11(i)=1/(1+exp(-H11(i)));
        OUTH12(i)=1/(1+exp(-H12(i)));
        OUTH13(i)=1/(1+exp(-H13(i)));
        OUTH14(i)=1/(1+exp(-H14(i)));
    end
    
    for i = 1:20
        OUT(i)=OUTH11(i)*w21+OUTH12(i)*w22+OUTH13(i)*w23+OUTH14(i)*w24+b21;
        OUT(i)=0.12345*OUT(i);
        e(i)=F(i)-OUT(i);
    end
    E=0;
    
    for i = 1:20
        E=E+e(i)^2/2;
    end
        
end

%Analogiskai kaip figure(2) palyginam signalus
figure(3)
hold on
plot(X,F,'rx',X,OUT,'go');
plot(X,F,'b',X,OUT,'k');
hold off
title('Originalus signalas ir apmokyto tinklo signalas');
legend('Originalus','OUT');

